from src.model.mtb_trainer import MTBTrainer
from src.model.dataset import load_dataloaders
from transformers import TrainingArguments, BertModel, BertTokenizer
from argparse import ArgumentParser
import torch
import logging


def main():

    model_size = "bert-base-uncased"
    model = BertModel.from_pretrained(model_size)
    tokenizer = BertTokenizer.from_pretrained(model_size, do_lower_case=False)
    train_data_path = "./Vault/data/mtb_training/combined_train.txt"
    batch_size = 2
    base_path = "./Vault/mtb/mtb_training/"

    training_args = TrainingArguments(
        output_dir = "Vault/mtb/mtb_training/",
        gradient_accumulation_steps = 16,
        per_device_train_batch_size = batch_size,
        num_train_epochs = 1,
        save_strategy = "steps",
        save_steps = 1000
    )
    print("Establishing Trainer")
    trainer = MTBTrainer(model, training_args, tokenizer, train_data_path, base_path)
    print("Starting Training")
    trainer.train()
    print("Training Complete")


if __name__=="__main__":
    main()
