#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/plkmo/BERT-Relation-Extraction, full credit must be given to the original author (see below). Adaptations include, removing unncesseary calls to other bert models and adapting the entity recognition to include more entity types (particularly cardinals). Notes and comments are my own.

@author: weetee
"""
from src.preprocessing_funcs import load_dataloaders
from src.mtb_trainer import train_and_fit
import logging
from argparse import ArgumentParser

'''
This trains the BERT model on matching the blanks
'''

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--pretrain_data", type=str, default="./Vault/mtb/mtb_training/combined_train.txt", \
                        help="pre-training data .txt file path")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=8, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
    parser.add_argument("--num_epochs", type=int, default=10, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00003, help="learning rate")


    args = parser.parse_args()

    output = train_and_fit(args)
