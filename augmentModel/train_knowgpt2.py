import torch
import torch.nn as nn
import transformers
# import knowbert model
from src.model.model import KnowGPT2Model, KnowGPT2LMHeadModel
from transformers import GPT2LMHeadModel
from pynvml import *
# import knowledge bases
from src.knowledge.khangraph import KhanGraph
from src.datasets.khan_academy import KhanAcademyMathDataset
from src.datasets.mathematica_with_steps import MathematicaWithStepsMathDataset
from src.datasets.MATH import MATHDataset
# utils
import os
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from datetime import datetime

def load_data(dataroot, multiplier, mode):

    tokenizer = get_tokenizer_gpt()

    if mode == "pretrain":
        train_data = []

        khan_root = dataroot + "amps/khan"
        wolf_root = dataroot + "amps/mathematica"

        khan_multiplier = multiplier
        wolf_multiplier = multiplier * 0.1

        train_data.append(KhanAcademyMathDataset(
                dataroot=khan_root,
                tokenizer=tokenizer,
                max_tokens=1024,
                mode='gpt2',
                mode_answer='mixed_hints',
                len_multiplier=khan_multiplier,
                latex_mask=False))

        train_data.append(MathematicaWithStepsMathDataset(
                dataroot=wolf_root,
                tokenizer=tokenizer,
                max_tokens=1024,
                mode='gpt2',
                len_multiplier=wolf_multiplier))

        return torch.utils.data.ConcatDataset(train_data)

    elif mode == "finetune":

        train_data = []

        math_root = dataroot + "MATH/train"

        math_multiplier = 1

        train_data.append(MATHDataset(
            dataroot=math_root,
            tokenizer=tokenizer, # Set in run_training(), not in dataset creation
            max_tokens=1024,
            mode='gpt2',
            mode_answer="mixed_final_boxed_and_full",
            len_multiplier=math_multiplier,
            peek_fraction=(0.1, 1.0),
            packing=True,    # Special for fine-tuning
            randomize=True
        ))

        return torch.utils.data.ConcatDataset(train_data)

    else:
        print("Invalid training mode, must be either pretrain or finetune")

def get_tokenizer_gpt():
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    return tokenizer


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def save_output(path, output):
    log = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"./Vault/know_gpt2_output/{log}/log/"
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, "command.txt"), 'w') as out_command:
        f.write(output)


def predict(model, batch, device):
    # separate data and cached
    data, caches = batch[:4], batch[4:]
    input_ids, token_type_ids, next_sentence_labels, masked_labels = data
    # predict
    model.set_valid_kb_caches(*caches)
    return model.forward(
        input_ids=input_ids.to(device),
        token_type_ids=token_type_ids.to(device),
        labels=masked_labels.to(device),
        next_sentence_label=next_sentence_labels.to(device)
    )



class GPT2Trainer(transformers.Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            print("Making AdamW Optimizer")
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

        if self.lr_scheduler is None:

            if self.args.warmup_steps == -1:
                print("Using constant LR")
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda steps: 1.0)
            else:
                print("Using Linear warmup LR")
                self.lr_scheduler = self.get_linear_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
                )

    @staticmethod
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        """
        Linear warmup from 0 to max lr, then linear decay from max_lr to 0.1*max_lr
        As done in https://arxiv.org/pdf/2010.14701.pdf
        """
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            min_lr_multiplier = 0.1
            return max(
                min_lr_multiplier,
                ((1 - min_lr_multiplier) * float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))) + min_lr_multiplier
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def run_training(train_data, model):

    print_gpu_utilization()

    save_steps = len(train_data)
    save_steps = int(save_steps / torch.cuda.device_count())
    # Gradient accumulation steps
    save_steps = int(save_steps / 16)
    # Batch size per replica
    save_steps = int(save_steps / 2)



    print("Save Steps = ", save_steps)

    start_epoch = 0
    start_iteration = 0

    ## Dataloading ########################################################
    train_data.start_iteration = start_iteration

    ## Start Loop ########################################################
    print(f"Setting up Trainer")

    training_args = transformers.TrainingArguments(
        output_dir="/home/besperk/Code/knowledge-graph/Vault/know_gpt2_output/",
        overwrite_output_dir=False,

        do_train=True,
        do_eval=False,
        do_predict=True,
        evaluation_strategy='no',
        eval_steps=0,

        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,

        learning_rate=5e-5,
        weight_decay=0.05,
        warmup_steps=-1,
        max_grad_norm=100000.0, # Essentially disable gradient clipping

        logging_dir="/home/besperk/Code/knowledge-graph/Vault/know_gpt2_output/",
        logging_first_step=True,
        logging_steps=5,
        save_steps=save_steps,
        save_total_limit=10, # Only save the last epoch

        dataloader_drop_last=True,
        dataloader_num_workers=1,

        local_rank=-1,
        tpu_num_cores=None,
    )

    trainer = GPT2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )

    print(f"Clearing CUDA cache")
    # NOTE: Check how much space there is on the GPU
    print_gpu_utilization()
    torch.cuda.empty_cache()

    print(f"STARTING TRAINING. save_steps={save_steps}")
    trainer.train()

    # NOTE: Should save to output_dir from training arguments
    trainer.save_model()
    print("Finished")
    return


def main():

    print_gpu_utilization()

    # Prepare Arguments
    main_device = 'cuda:0'
    base_model = "gpt2"
    # For AMPS training load a pretrained GPT2 model or load the AMPS trained model for fine-tuning on MATH or None for a vanilla GPT2 model from transformers
    load = None
    # The path to the knowledge base for training a KnowGPT2Model, otherwise leave as None
    data_path = None
    data_root = "/home/besperk/Code/MATH-Data/"
    multiplier = 1

    print("Loading Model... (%s)" % base_model)

    # Prepare Model
    if load:
        print("Loading a pretrained KnowGPT2Model")
        model = KnowGPT2LMHeadModel.from_pretrained(load)
    else:
        print("Loading a vanilla GPTModel")
        model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Note: once a KnowGPT2Model has been trained, we don't want to try and add another kb at the same layer
    if data_path:
        kb = model.add_kb(10, KhanGraph(data_path=data_path))
        model.freeze_layers(10)

    model.to(main_device)

    # Get Training data
    train_data = load_data(data_root, multiplier, "pretrain")

    run_training(train_data, model)
    return


if __name__ == "__main__":
    main()
