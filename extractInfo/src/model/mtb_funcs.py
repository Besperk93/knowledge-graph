"""
## NOTE ##:
Adapted from weetee to measure the loss across both the Masked Language Model (Cloze task) and the Match The Blanks task. Full credit given to the author. Main tweaks are adjusting the script to only use the bert model and to detect numerical entities which were previously ignored
####
Created on Tue Nov 26 18:12:22 2019
@author: weetee
"""

import os
import math
import torch
import torch.nn as nn
from itertools import combinations
from src.utilities import save_as_pickle, load_pickle
import logging
from tqdm import tqdm

class Two_Headed_Loss(nn.Module):
    '''
    Implements LM Loss and matching-the-blanks loss concurrently
    '''
    def __init__(self, lm_ignore_idx, use_logits=False, normalize=False):
        super(Two_Headed_Loss, self).__init__()
        self.lm_ignore_idx = lm_ignore_idx
        self.LM_criterion = nn.CrossEntropyLoss(ignore_index=self.lm_ignore_idx)
        self.use_logits = use_logits
        self.normalize = normalize

        if not self.use_logits:
            self.BCE_criterion = nn.BCELoss(reduction='mean')
        else:
            self.BCE_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def p_(self, f1_vec, f2_vec):
        if self.normalize:
            factor = 1/(torch.norm(f1_vec)*torch.norm(f2_vec))
        else:
            factor = 1.0

        if not self.use_logits:
            p = 1/(1 + torch.exp(-factor*torch.dot(f1_vec, f2_vec)))
        else:
            p = factor*torch.dot(f1_vec, f2_vec)
        return p

    def dot_(self, f1_vec, f2_vec):
        return -torch.dot(f1_vec, f2_vec)

    def forward(self, lm_logits, blank_logits, lm_labels, blank_labels, verbose=False):
        '''
        lm_logits: (batch_size, sequence_length, hidden_size)
        lm_labels: (batch_size, sequence_length, label_idxs)
        blank_logits: (batch_size, embeddings)
        blank_labels: (batch_size, 0 or 1)
        '''
        pos_idxs = [i for i, l in enumerate(blank_labels.squeeze().tolist()) if l == 1]
        neg_idxs = [i for i, l in enumerate(blank_labels.squeeze().tolist()) if l == 0]

        if len(pos_idxs) > 1:
            # positives
            pos_logits = []
            for pos1, pos2 in combinations(pos_idxs, 2):
                pos_logits.append(self.p_(blank_logits[pos1, :], blank_logits[pos2, :]))
            pos_logits = torch.stack(pos_logits, dim=0)
            pos_labels = [1.0 for _ in range(pos_logits.shape[0])]
        else:
            pos_logits, pos_labels = torch.FloatTensor([]), []
            if blank_logits.is_cuda:
                pos_logits = pos_logits.cuda()

        # negatives
        neg_logits = []
        for pos_idx in pos_idxs:
            for neg_idx in neg_idxs:
                neg_logits.append(self.p_(blank_logits[pos_idx, :], blank_logits[neg_idx, :]))
        neg_logits = torch.stack(neg_logits, dim=0)
        neg_labels = [0.0 for _ in range(neg_logits.shape[0])]

        blank_labels_ = torch.FloatTensor(pos_labels + neg_labels)

        if blank_logits.is_cuda:
            blank_labels_ = blank_labels_.cuda()

        lm_loss = self.LM_criterion(lm_logits, lm_labels)

        blank_loss = self.BCE_criterion(torch.cat([pos_logits, neg_logits], dim=0), \
                                        blank_labels_)

        if verbose:
            print("LM loss, blank_loss for last batch: %.5f, %.5f" % (lm_loss, blank_loss))

        total_loss = lm_loss + blank_loss
        return total_loss



def evaluate_(lm_logits, blanks_logits, masked_for_pred, blank_labels, tokenizer, print_=True):
    '''
    evaluate must be called after loss.backward()
    '''
    # lm_logits
    lm_logits_pred_ids = torch.softmax(lm_logits, dim=-1).max(1)[1]
    lm_accuracy = ((lm_logits_pred_ids == masked_for_pred).sum().float()/len(masked_for_pred)).item()

    if print_:
        print("Predicted masked tokens: \n")
        print(tokenizer.decode(lm_logits_pred_ids.cpu().numpy() if lm_logits_pred_ids.is_cuda else \
                               lm_logits_pred_ids.numpy()))
        print("\nMasked labels tokens: \n")
        print(tokenizer.decode(masked_for_pred.cpu().numpy() if masked_for_pred.is_cuda else \
                               masked_for_pred.numpy()))

    '''
    # blanks
    blanks_diff = ((blanks_logits - blank_labels)**2).detach().cpu().numpy().sum() if blank_labels.is_cuda else\
                    ((blanks_logits - blank_labels)**2).detach().numpy().sum()
    blanks_mse = blanks_diff/len(blank_labels)

    if print_:
        print("Blanks MSE: ", blanks_mse)
    '''
    blanks_mse = 0
    return lm_accuracy, blanks_mse
