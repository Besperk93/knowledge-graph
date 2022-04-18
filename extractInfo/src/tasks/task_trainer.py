#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/plkmo/BERT-Relation-Extraction, full credit must be given to the original author (see below). Adaptations include, removing unncesseary calls to other bert models and adapting the entity recognition to include more entity types (particularly cardinals). Notes and comments are my own.

@author: weetee
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .task_funcs import load_state, load_results, evaluate_, evaluate_results
from ..utilities import save_as_pickle, load_pickle
import matplotlib.pyplot as plt
import time
import logging
# NOTE: Get rid of alternative models as just using Bert, which can be imported from transformers directly
from transformers import BertModel, BertTokenizer

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

######### DataLoaders #########

def load_dataloaders(args):

    model = 'bert-base-uncased'
    lower_case = True
    model_name = 'BERT'

    # NOTE: Check to see id there's a saved tokenizer to use
    if os.path.isfile("./Vault/mtb/output/%s_tokenizer.pkl" % model_name):
        tokenizer = load_pickle("%s_tokenizer.pkl" % model_name)
        logger.info("Loaded tokenizer from pre-trained blanks model")
    else:
        logger.info("Pre-trained blanks tokenizer not found, initializing new tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=False)
        # NOTE: Add Match The Blank special tokens
        tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'])
        # NOTE: Save the tokenizer to be used again
        save_as_pickle("%s_tokenizer.pkl" % model_name, tokenizer)
        logger.info("Saved %s tokenizer at ./Vault/mtb/output/%s_tokenizer.pkl" %(model_name, model_name))

    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    assert e1_id != e2_id != 1

    if args.task == 'semeval':
        relations_path = './Vault/mtb/output/relations.pkl'
        train_path = './Vault/mtb/output/df_train.pkl'
        test_path = './Vault/mtb/output/df_test.pkl'
        if os.path.isfile(relations_path) and os.path.isfile(train_path) and os.path.isfile(test_path):
            rm = load_pickle('relations.pkl')
            df_train = load_pickle('df_train.pkl')
            df_test = load_pickle('df_test.pkl')
            logger.info("Loaded preproccessed data.")
        else:
            df_train, df_test, rm = preprocess_semeval2010_8(args)

        train_set = semeval_dataset(df_train, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
        test_set = semeval_dataset(df_test, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
        train_length = len(train_set); test_length = len(test_set)
        PS = Pad_Sequence(seq_pad_value=tokenizer.pad_token_id,\
                          label_pad_value=tokenizer.pad_token_id,\
                          label2_pad_value=-1)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, \
                                  num_workers=0, collate_fn=PS, pin_memory=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, \
                                  num_workers=0, collate_fn=PS, pin_memory=False)
    elif args.task == 'fewrel':
        df_train, df_test = preprocess_fewrel(args, do_lower_case=lower_case)
        train_loader = fewrel_dataset(df_train, tokenizer=tokenizer, seq_pad_value=tokenizer.pad_token_id,
                                      e1_id=e1_id, e2_id=e2_id)
        train_length = len(train_loader)
        test_loader, test_length = None, None

    return train_loader, test_loader, train_length, test_length


######### Traniner #########

def train_and_fit(args):

    # Check for AMP
    if args.fp16:
        from apex import amp
    else:
        amp = None

    # Check for CUDA
    cuda = torch.cuda.is_available()

    # These are coming from PyTorch Dataloaders and are specifc to the task datasets
    train_loader, test_loader, train_len, test_len = load_dataloaders(args)
    logger.info("Loaded %d Training samples." % train_len)


    model = args.model_size #'bert-base-uncased'
    lower_case = True
    model_name = 'BERT'
    net = BertModel.from_pretrained(model, force_download=False, \
                            model_size=args.model_size,
                            task='classification' if args.task != 'fewrel' else 'fewrel',\
                            n_classes_=args.num_classes)

    # NOTE: Tokenizer will be saved at this location when loading dataloaders above
    tokenizer = load_pickle("%s_tokenizer.pkl" % model_name)
    net.resize_token_embeddings(len(tokenizer))
    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    assert e1_id != e2_id != 1

    if cuda:
        net.cuda()

    logger.info("FREEZING MOST HIDDEN LAYERS...")

    # NOTE: These layers will be trained
    unfrozen_layers = ["classifier", "pooler", "encoder.layer.11", \
                       "classification_layer", "blanks_linear", "lm_linear", "cls"]

    # NOTE: Freezing the relevant layers
    for name, param in net.named_parameters():
        if not any([layer in name for layer in unfrozen_layers]):
            print("[FROZE]: %s" % name)
            param.requires_grad = False
        else:
            print("[FREE]: %s" % name)
            param.requires_grad = True

    # NOTE: Loading a saved model, pretrained on match the blanks
    if args.use_pretrained_blanks == 1:
        logger.info("Loading model pre-trained on blanks at ./Vault/mtb/output/test_checkpoint.pth.tar...")
        checkpoint_path = "./Vault/mtb/output/test_checkpoint.pth.tar"
        checkpoint = torch.load(checkpoint_path)
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        net.load_state_dict(pretrained_dict, strict=False)
        # NOTE: Delete these variables to stop them taking up space
        del checkpoint, pretrained_dict, model_dict

    # NOTE: Check these match up with FewRel task
    # NOTE: are they in any way different from the hf defaults?
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam([{"params":net.parameters(), "lr": args.lr}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,24,26,30], gamma=0.8)

    start_epoch, best_pred, amp_checkpoint = load_state(net, optimizer, scheduler, args, load_best=False)

    # NOTE: If AMP is available, use it to speed up training
    if (args.fp16) and (amp is not None):
        logger.info("Using fp16...")
        net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
        if amp_checkpoint is not None:
            amp.load_state_dict(amp_checkpoint)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,24,26,30], gamma=0.8)

    losses_per_epoch, accuracy_per_epoch, test_f1_per_epoch = load_results()

    logger.info("Starting training process...")
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    update_size = len(train_loader)//10
    # NOTE: This is the training loop
    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        # NOTE: Here is the call to train()
        net.train(); total_loss = 0.0; losses_per_batch = []; total_acc = 0.0; accuracy_per_batch = []
        for i, data in enumerate(train_loader, 0):

            ####### Normal BERT Training #######
            x, e1_e2_start, labels, _,_,_ = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

            # NOTE: Move to GPU if available
            if cuda:
                x = x.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()

            # NOTE: Get results from the model during training
            classification_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
                          e1_e2_start=e1_e2_start)

            #return classification_logits, labels, net, tokenizer # for debugging now

            # NOTE: This looks to be the same as in the default trainer
            loss = criterion(classification_logits, labels.squeeze(1))
            loss = loss/args.gradient_acc_steps

            # NOTE: Same in both
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # NOTE: Same in both
            if args.fp16:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
            else:
                # NOTE: in the default trainer this looks similar but calls a function from the hf model, rather than a pytorch function
                grad_norm = clip_grad_norm_(net.parameters(), args.max_norm)

            # NOTE: Same in both
            if (i % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()

            # NOTE: Difference between classification logits and lm_logits
            total_loss += loss.item()
            total_acc += evaluate_(classification_logits, labels, \
                                   ignore_idx=-1)[0]

            if (i % update_size) == (update_size - 1):
                losses_per_batch.append(args.gradient_acc_steps*total_loss/update_size)
                accuracy_per_batch.append(total_acc/update_size)
                print('[Epoch: %d, %5d/ %d points] total loss, accuracy per batch: %.3f, %.3f' %
                      (epoch + 1, (i + 1)*args.batch_size, train_len, losses_per_batch[-1], accuracy_per_batch[-1]))
                total_loss = 0.0; total_acc = 0.0

            ###################################

        scheduler.step()
        # NOTE: This is extracting results from each step
        results = evaluate_results(net, test_loader, pad_id, cuda)
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        accuracy_per_epoch.append(sum(accuracy_per_batch)/len(accuracy_per_batch))
        test_f1_per_epoch.append(results['f1'])
        print("Epoch finished, took %.2f seconds." % (time.time() - start_time))
        print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
        print("Train accuracy at Epoch %d: %.7f" % (epoch + 1, accuracy_per_epoch[-1]))
        print("Test f1 at Epoch %d: %.7f" % (epoch + 1, test_f1_per_epoch[-1]))

        # Check and save the best prediction after each step
        if accuracy_per_epoch[-1] > best_pred:
            best_pred = accuracy_per_epoch[-1]
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': accuracy_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                    'amp': amp.state_dict() if amp is not None else amp
                }, os.path.join("./Vault/mtb/output/" , "task_test_model_best.pth.tar"))

        if (epoch % 1) == 0:
            # Save full info after each epoch (full run through train data)
            save_as_pickle("task_test_losses_per_epoch.pkl", losses_per_epoch)
            save_as_pickle("task_train_accuracy_per_epoch.pkl", accuracy_per_epoch)
            save_as_pickle("task_test_f1_per_epoch.pkl", test_f1_per_epoch)
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': accuracy_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                    'amp': amp.state_dict() if amp is not None else amp
                }, os.path.join("./Vault/mtb/output/" , "task_test_checkpoint.pth.tar"))

    # Create graphs
    # NOTE: this plot shows loss per epoch
    logger.info("Finished Training!")
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    ax.scatter([e for e in range(len(losses_per_epoch))], losses_per_epoch)
    ax.tick_params(axis="both", length=2, width=1, labelsize=14)
    ax.set_xlabel("Epoch", fontsize=22)
    ax.set_ylabel("Training Loss per batch", fontsize=22)
    ax.set_title("Training Loss vs Epoch", fontsize=32)
    plt.savefig(os.path.join("./Vault/mtb/output/" ,"task_loss_vs_epoch.png"))

    # NOTE: This plot shows accuracy per epoch
    fig2 = plt.figure(figsize=(20,20))
    ax2 = fig2.add_subplot(111)
    ax2.scatter([e for e in range(len(accuracy_per_epoch))], accuracy_per_epoch)
    ax2.tick_params(axis="both", length=2, width=1, labelsize=14)
    ax2.set_xlabel("Epoch", fontsize=22)
    ax2.set_ylabel("Training Accuracy", fontsize=22)
    ax2.set_title("Training Accuracy vs Epoch", fontsize=32)
    plt.savefig(os.path.join("./Vault/mtb/output/" ,"task_train_accuracy_vs_epoch_%d.png"))

    # NOTE: This plot shows f1 per epoch
    fig3 = plt.figure(figsize=(20,20))
    ax3 = fig3.add_subplot(111)
    ax3.scatter([e for e in range(len(test_f1_per_epoch))], test_f1_per_epoch)
    ax3.tick_params(axis="both", length=2, width=1, labelsize=14)
    ax3.set_xlabel("Epoch", fontsize=22)
    ax3.set_ylabel("Test F1 Accuracy", fontsize=22)
    ax3.set_title("Test F1 vs Epoch", fontsize=32)
    plt.savefig(os.path.join("./Vault/mtb/output/" ,"task_test_f1_vs_epoch.png"))

    return net
