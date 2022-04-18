#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/plkmo/BERT-Relation-Extraction, full credit must be given to the original author (see below). Adaptations include, removing unncesseary calls to other bert models and adapting the entity recognition to include more entity types (particularly cardinals). Notes and comments are my own.

@author: weetee
"""
from src.tasks.task_trainer import train_and_fit
from src.tasks.task_eval import infer_from_trained, FewRel
import logging
from argparse import ArgumentParser

'''
This fine-tunes the BERT model on SemEval, FewRel tasks
'''

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default='semeval', help='semeval, fewrel')
    parser.add_argument("--train_data", type=str, default='./Vault/mtb/mtb_eval/semeval2010_task8/TRAIN_FILE.TXT',
                        help="training data .txt file path")
    parser.add_argument("--test_data", type=str, default='./Vault/mtb/mtb_eval/semeval2010_task8/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT', \
                        help="test data .txt file path")
    parser.add_argument("--use_pretrained_blanks", type=int, default=1, help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model")
    parser.add_argument("--num_classes", type=int, default=19, help='number of relation classes')
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
    parser.add_argument("--num_epochs", type=int, default=10, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00003, help="learning rate")
    parser.add_argument("--train", type=int, default=1, help="0: Don't train, 1: train")
    parser.add_argument("--infer", type=int, default=1, help="0: Don't infer, 1: Infer")

    args = parser.parse_args()

    if (args.train == 1) and (args.task != 'fewrel'):
        net = train_and_fit(args)

    if (args.infer == 1) and (args.task != 'fewrel'):
        inferer = infer_from_trained(args, detect_entities=True)
        test = "The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor."
        inferer.infer_sentence(test, detect_entities=False)
        test2 = "After eating the chicken, he developed a sore throat the next morning."
        inferer.infer_sentence(test2, detect_entities=True)

        while True:
            sent = input("Type input sentence ('quit' or 'exit' to terminate):\n")
            if sent.lower() in ['quit', 'exit']:
                break
            inferer.infer_sentence(sent, detect_entities=False)

    if args.task == 'fewrel':
        fewrel = FewRel(args)
        meta_input, e1_e2_start, meta_labels, outputs = fewrel.evaluate()
        # Save outputs here for reference
        with open("./Vault/mtb/output/fewrel_meta_input.txt", 'w') as out_file:
            json.dump(meta_input, out_file)
        with open("./Vault/mtb/output/fewrel_e1_e2_start.txt", 'w') as out_file:
            json.dump(meta_input, out_file)
        with open("./Vault/mtb/output/fewrel_meta_labels.txt", 'w') as out_file:
            json.dump(meta_input, out_file)
        with open("./Vault/mtb/output/fewrel_outputs.txt", 'w') as out_file:
            json.dump(meta_input, out_file)
