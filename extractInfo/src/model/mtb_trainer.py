from transformers import Trainer, TrainingArguments, TrainOutput, BertModel
from mtb_model import Two_Headed_Loss, evaluate_
from preprocessing_funcs import load_dataloaders
from .utilities import save_as_pickle, load_pickle
import torch
import torch.nn as nn


class MTBTrainer(Trainer):

    def __init__(self, config):
        # Handle the init for parent class plus any extras
        super(MTBTrainer, self).__init__(*config)
        self.training_args = config.args
        # Extras
        self.accuracy_per_epoch = []
        self.loss_per_epoch = []
        self.f1_per_epoch = []

    def train(self):
        # Alter the training loop for the MTB task from Trainer (1103-1583)
        args = self.training_args

        # Check if CUDA
        cuda = torch.cuda.is_available()

        # Get Dataloaders
        # [Trainer] train_dataloader = self.get_train_dataloader()
        train_loader = load_dataloaders(args)
        train_len = len(train_loader)
        logger.info("Loaded %d pre-training samples." % train_len)

        # Get Model and tokenizer
        model = self.model
        tokenizer = self.tokenizer
        net.resize_token_embeddings(len(tokenizer))
        # NOTE: MTB specific, keep
        e1_id = tokenizer.convert_tokens_to_ids('[E1]')
        e2_id = tokenizer.convert_tokens_to_ids('[E2]')
        assert e1_id != e2_id != 1

        # Move model to GPU if available
        if cuda:
            net.cuda()

        # Freeze layers if required
        if args.freeze == 1:
            logger.info("FREEZING MOST HIDDEN LAYERS...")
            unfrozen_layers = ["classifier", "pooler", "encoder.layer.11", "encoder.layer.10",          "encoder.layer.9", "blanks_linear", "lm_linear", "cls"]

            for name, param in net.named_parameters():
                if not any([layer in name for layer in unfrozen_layers]):
                    print("[FROZE]: %s" % name)
                    param.requires_grad = False
                else:
                    print("[FREE]: %s" % name)
                    param.requires_grad = True

        # Load custom loss model for MLM and MTB tasks
        criterion = Two_Headed_Loss(lm_ignore_idx=tokenizer.pad_token_id, use_logits=True, normalize=False)

        # Get optimizer and lr_scheduler
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        start_epoch, best_pred, amp_checkpoint = self.load_state(net, optimizer, scheduler, 0, load_best=False)

        losses_per_epoch, accuracy_per_epoch = self.load_results(0)

        # Output Training config to console
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        pad_id = tokenizer.pad_token_id
        mask_id = tokenizer.mask_token_id
        update_size = len(train_loader)//10

        ########## Training Loop ##########
        for epoch in range(start_epoch, args.num_train_epochs):
            start_time = time.time()
            net.train()
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

                blanks_logits, lm_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None, e1_e2_start=e1_e2_start)

                lm_logits = lm_logits[(x == mask_id)]
                if (i % update_size) == (update_size - 1):
                    verbose = True
                else:
                    verbose = False

                loss = criterion(lm_logits, blanks_logits, masked_for_pred, blank_labels, verbose=verbose)
                loss = loss/args.gradient_accumulation_steps
                loss.backward()

                grad_norm = clip_grad_norm_(net.parameters(), args.max_grad_norm)

                if (i % args.gradient_accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                total_acc += evaluate_(lm_logits, blanks_logits, masked_for_pred, blank_labels, tokenizer, print_=False)[0]

                if (i % update_size) == (update_size - 1):
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
                        'state_dict': net.state_dict(),\
                        'best_acc': accuracy_per_epoch[-1],\
                        'optimizer' : optimizer.state_dict(),\
                        'scheduler' : scheduler.state_dict(),\
                        'amp': amp.state_dict() if amp is not None else amp
                    }, os.path.join("./data/" , "test_model_best_%d.pth.tar" % 0))

            if (epoch % 1) == 0:
                save_as_pickle("test_losses_per_epoch_%d.pkl" % 0, losses_per_epoch)
                save_as_pickle("test_accuracy_per_epoch_%d.pkl" % 0, accuracy_per_epoch)
                torch.save({
                        'epoch': epoch + 1,\
                        'state_dict': net.state_dict(),\
                        'best_acc': accuracy_per_epoch[-1],\
                        'optimizer' : optimizer.state_dict(),\
                        'scheduler' : scheduler.state_dict(),\
                        'amp': amp.state_dict() if amp is not None else amp
                    }, os.path.join("./data/" , "test_checkpoint_%d.pth.tar" % 0))

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
        ax.set_title("Training Loss vs Epoch", fontsize=32)
        plt.savefig(os.path.join("./data/" ,"loss_vs_epoch_%d.png" % 0))

        # Create plot of accuracy per epoch
        fig2 = plt.figure(figsize=(20,20))
        ax2 = fig2.add_subplot(111)
        ax2.scatter([e for e in range(len(accuracy_per_epoch))], accuracy_per_epoch)
        ax2.tick_params(axis="both", length=2, width=1, labelsize=14)
        ax2.set_xlabel("Epoch", fontsize=22)
        ax2.set_ylabel("Test Masked LM Accuracy", fontsize=22)
        ax2.set_title("Test Masked LM Accuracy vs Epoch", fontsize=32)
        plt.savefig(os.path.join("./data/" ,"accuracy_vs_epoch_%d.png" % 0))
        logger.info("Plots Saved in ./data/")


    def load_state(self, net, optimizer, scheduler, load_best=False):
        """ Loads saved model and optimizer states if exists """
        base_path = "./data/"
        amp_checkpoint = None
        checkpoint_path = os.path.join(base_path,"test_checkpoint_%d.pth.tar" % 0)
        best_path = os.path.join(base_path,"test_model_best_%d.pth.tar" % 0)
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
            net.load_state_dict(checkpoint['state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            amp_checkpoint = checkpoint['amp']
            logger.info("Loaded model and optimizer.")
        return start_epoch, best_pred, amp_checkpoint

    def load_results(self):
        """ Loads saved results if exists """
        losses_path = "./data/test_losses_per_epoch_%d.pkl" % 0
        accuracy_path = "./data/test_accuracy_per_epoch_%d.pkl" % 0
        if os.path.isfile(losses_path) and os.path.isfile(accuracy_path):
            losses_per_epoch = load_pickle("test_losses_per_epoch_%d.pkl" % model_no)
            accuracy_per_epoch = load_pickle("test_accuracy_per_epoch_%d.pkl" % model_no)
            logger.info("Loaded results buffer")
        else:
            losses_per_epoch, accuracy_per_epoch = [], []
        return losses_per_epoch, accuracy_per_epoch
