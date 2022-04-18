from transformers import Trainer, TrainingArguments, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.bert.configuration_bert import BertConfig
from src.model.mtb_funcs import Two_Headed_Loss, evaluate_
from src.model.dataset import load_dataloaders
from src.utilities import save_as_pickle, load_pickle
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import logging
import os
import time

tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

class MTBTrainer(Trainer):

    def __init__(self, model, args, tokenizer, data_path, base_path):
        # Handle the init for parent class plus any extras
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.base_path = base_path
        self.activation = nn.Tanh()
        self.config = BertConfig()
        self.cls = BertOnlyMLMHead(self.config)
        super(MTBTrainer, self).__init__(model=self.model, args=self.args, tokenizer=self.tokenizer)
        # Extras
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer)
        self.accuracy_per_epoch = []
        self.loss_per_epoch = []
        self.f1_per_epoch = []


    def train(self):
        # Alter the training loop for the MTB task from Trainer (1103-1583)
        args = self.args

        # Check if CUDA
        cuda = torch.cuda.is_available()

        # Get Dataloaders
        # [Trainer] train_dataloader = self.get_train_dataloader()
        try:
            train_loader = load_dataloaders(args, self.data_path)
        except Exception as e:
            print(f"Error loading dataset: {repr(e)}")
            return
        train_len = len(train_loader)
        logger.info("Loaded %d pre-training samples." % train_len)

        # Get Model and tokenizer
        model = self.model
        try:
            tokenizer = load_pickle("/home/besperk/Code/knowledge-graph/Vault/mtb/mtb_training/BERT_tokenizer.pkl")
            print("Loading saved tokenizer")
        except:
            print("Setting up new tokenizer")
            tokenizer = self.tokenizer
            tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'])

        model.resize_token_embeddings(len(tokenizer))
        # NOTE: MTB specific, keep
        e1_id = tokenizer.convert_tokens_to_ids('[E1]')
        e2_id = tokenizer.convert_tokens_to_ids('[E2]')
        assert e1_id != e2_id != 1

        # Move model to GPU if available
        if cuda:
            model.cuda()
            self.cls.cuda()

        optimizer = self.optimizer
        scheduler = self.scheduler

        # Load custom loss model for MLM and MTB tasks
        criterion = Two_Headed_Loss(lm_ignore_idx=tokenizer.pad_token_id, use_logits=True, normalize=False)

        # Get optimizer and lr_scheduler
        training_steps = args.num_train_epochs * len(train_loader)
        # self.create_optimizer_and_scheduler(training_steps)

        start_epoch, best_pred, amp_checkpoint = self.load_state(model, self.optimizer, self.scheduler, load_best=False)

        losses_per_epoch, accuracy_per_epoch = self.load_results()

        pad_id = tokenizer.pad_token_id
        mask_id = tokenizer.mask_token_id
        update_size = len(train_loader)//10

        print(f"Beggining Training")
        print(f"Training Steps: {training_steps}")
        print(f"Update Size: {update_size}")
        print(f"Args: {args}")

        ########## Training Loop ##########
        for epoch in tqdm(range(start_epoch, args.num_train_epochs)):
            start_time = time.time()
            model.train()
            total_loss = 0.0
            losses_per_batch = []
            total_acc = 0.0
            lm_accuracy_per_batch = []
            for i, data in enumerate(train_loader, 0):
                ########### MTB ############
                x, masked_for_pred, e1_e2_start, _, blank_labels, _,_,_,_,_ = data
                masked_for_pred1 =  masked_for_pred
                masked_for_pred = masked_for_pred[(masked_for_pred != pad_id)]
                if masked_for_pred.shape[0] == 0:
                    print('Empty dataset, skipping...')
                    continue
                attention_mask = (x != pad_id).float()
                token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

                if cuda:
                    x = x.cuda(); masked_for_pred = masked_for_pred.cuda()
                    attention_mask = attention_mask.cuda()
                    token_type_ids = token_type_ids.cuda()

                ###### MTB Training Logits ######
                # NOTE: Can get the info from standard BERT output rather than creating a custom BERT model
                pooled_outputs = model(x, token_type_ids=token_type_ids, attention_mask=attention_mask)

                sequence_output = pooled_outputs.last_hidden_state
                blankv1v2 = sequence_output[:, e1_e2_start, :]
                buffer = []
                for i in range(blankv1v2.shape[0]): # iterate batch & collect
                    v1v2 = blankv1v2[i, i, :, :]
                    v1v2 = torch.cat((v1v2[0], v1v2[1]))
                    buffer.append(v1v2)
                del blankv1v2
                v1v2 = torch.stack([a for a in buffer], dim=0)
                del buffer

                # NOTE: There are different activations for the logits for the other tasks (FewRel, SemEval etc) here and some attributes that need to be set in the model (activation, cls etc)

                # NOTE: For Matching the blanks
                blanks_logits = self.activation(v1v2)
                # NOTE: For the standard MLM
                lm_logits = self.cls(sequence_output)

                ##########################

                lm_logits = lm_logits[(x == mask_id)]

                loss = criterion(lm_logits, blanks_logits, masked_for_pred, blank_labels, verbose=False)
                loss = loss/args.gradient_accumulation_steps
                loss.backward()

                grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)


                optimizer.step()
                optimizer.zero_grad()



                total_loss += loss.item()
                total_acc += evaluate_(lm_logits, blanks_logits, masked_for_pred, blank_labels, tokenizer, print_=False)[0]

                if (i % 4) == 0:
                losses_per_batch.append(args.gradient_accumulation_steps*total_loss/update_size)
                lm_accuracy_per_batch.append(total_acc/update_size)
                print('[Epoch: %d, %5d/ %d points] total loss, lm accuracy per batch: %.3f, %.3f' %
                      (epoch + 1, (i + 1), train_len, losses_per_batch[-1], lm_accuracy_per_batch[-1]))
                total_loss = 0.0; total_acc = 0.0
                logger.info("Last batch samples (pos, neg): %d, %d" % ((blank_labels.squeeze() == 1).sum().item(), (blank_labels.squeeze() == 0).sum().item()))


                ##############################

            scheduler.step()
            losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
            accuracy_per_epoch.append(sum(lm_accuracy_per_batch)/len(lm_accuracy_per_batch))
            print("Epoch finished, took %.2f seconds." % (time.time() - start_time))
            print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
            print("Accuracy at Epoch %d: %.7f" % (epoch + 1, accuracy_per_epoch[-1]))

            if accuracy_per_epoch[-1] > best_pred:
                best_pred = accuracy_per_epoch[-1]
                torch.save({
                        'epoch': epoch + 1,\
                        'state_dict': model.state_dict(),\
                        'best_acc': accuracy_per_epoch[-1],\
                        'optimizer' : self.optimizer.state_dict(),\
                        'scheduler' : self.scheduler.state_dict(),\
                        'amp': amp.state_dict() if amp is not None else amp
                    }, os.path.join(self.base_path , "test_model_best_%d.pth.tar" % 0))

            if (epoch % 1) == 0:
                save_as_pickle("test_losses_per_epoch_%d.pkl" % 0, losses_per_epoch)
                save_as_pickle("test_accuracy_per_epoch_%d.pkl" % 0, accuracy_per_epoch)
                torch.save({
                        'epoch': epoch + 1,\
                        'state_dict': model.state_dict(),\
                        'best_acc': accuracy_per_epoch[-1],\
                        'optimizer' : self.optimizer.state_dict(),\
                        'scheduler' : self.scheduler.state_dict(),\
                        'amp': amp.state_dict() if amp is not None else amp
                    }, os.path.join(self.base_path , "test_checkpoint_%d.pth.tar" % 0))

            # NOTE: Returns a named tuple, will this hold the info to create the graphs? (global_step, training_loss, metrics[])
            logger.info("Finished Training!")
            return plot_results(losses_per_epoch, accuracy_per_epoch)


    def plot_results(losses_per_epoch, accuracy_per_epoch):
        """Create and save plots of results"""
        # Create plot of loss per epoch
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(111)
        ax.scatter([e for e in range(len(losses_per_epoch))], losses_per_epoch)
        ax.tick_params(axis="both", length=2, width=1, labelsize=14)
        ax.set_xlabel("Epoch", fontsize=22)
        ax.set_ylabel("Training Loss per batch", fontsize=22)
        ax.set_ylim(bottom=0)
        ax.set_title("Training Loss vs Epoch", fontsize=32)
        plt.savefig(os.path.join(self.base_path ,"loss_vs_epoch_%d.png" % 0))

        # Create plot of accuracy per epoch
        fig2 = plt.figure(figsize=(20,20))
        ax2 = fig2.add_subplot(111)
        ax2.scatter([e for e in range(len(accuracy_per_epoch))], accuracy_per_epoch)
        ax2.tick_params(axis="both", length=2, width=1, labelsize=14)
        ax2.set_xlabel("Epoch", fontsize=22)
        ax2.set_ylabel("Test Masked LM Accuracy", fontsize=22)
        ax.set_ylim(bottom=0)
        ax2.set_title("Test Masked LM Accuracy vs Epoch", fontsize=32)
        plt.savefig(os.path.join(self.base_path ,"accuracy_vs_epoch_%d.png" % 0))
        logger.info("Plots Saved in ./data/")


    def load_state(self, model, optimizer, scheduler, load_best=False):
        """ Loads saved model and optimizer states if exists """
        amp_checkpoint = None
        checkpoint_path = os.path.join(self.base_path, "test_checkpoint_%d.pth.tar" % 0)
        best_path = os.path.join(self.base_path, "test_model_best_%d.pth.tar" % 0)
        start_epoch, best_pred, checkpoint = 0, 0, None
        if (load_best == True) and os.path.isfile(best_path):
            checkpoint = torch.load(best_path)
            logger.info("Loaded best model.")
        elif os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            logger.info("Loaded checkpoint model.")
        if checkpoint != None:
            start_epoch = checkpoint['epoch']
            best_pred = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            amp_checkpoint = checkpoint['amp']
            logger.info("Loaded model and optimizer.")
        return start_epoch, best_pred, amp_checkpoint

    def load_results(self):
        """ Loads saved results if exists """
        losses_path = os.path.join(self.base_path, "test_losses_per_epoch_%d.pkl" % 0)
        accuracy_path = os.path.join(self.base_path, "test_accuracy_per_epoch_%d.pkl" % 0)
        if os.path.isfile(losses_path) and os.path.isfile(accuracy_path):
            losses_per_epoch = load_pickle(os.path.join(self.base_path, "test_losses_per_epoch_%d.pkl" % model_no))
            accuracy_per_epoch = load_pickle(os.path.join(self.base_path, "test_accuracy_per_epoch_%d.pkl" % model_no))
            logger.info("Loaded results buffer")
        else:
            losses_per_epoch, accuracy_per_epoch = [], []
        return losses_per_epoch, accuracy_per_epoch
