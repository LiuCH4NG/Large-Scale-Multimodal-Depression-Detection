#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""
#----------------------------------------------------------------
# imports
#----------------------------------------------------------------
import torch
import numpy as np
from termcolor import colored

def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG :", 'green') + colored(msg, mcolor))

# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=15, delta=0, verbose=False, save_path=None):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.save_path:
            torch.save(model.state_dict(), self.save_path)
        if self.verbose:
            print(f"Validation loss decreased to {val_loss:.6f}. Saving model...")
        self.best_model = model.state_dict()

    def load_best_model(self, model):
        if self.best_model is not None:
            model.load_state_dict(self.best_model)


def collate_fn(batch):
    # Assuming x is the data and y is the label
    data, labels = zip(*batch)
    
    # Determine max length in batch
    max_length = max(d.shape[0] for d in data)
    
    # Pad data efficiently
    padded_data = np.array([
        np.pad(d, ((0, max_length - d.shape[0]), (0, 0)), mode='constant')
        for d in data
    ])  # Now it's a single NumPy array

    # Convert to PyTorch tensors
    return torch.from_numpy(padded_data).float(), torch.tensor(labels, dtype=torch.long)

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    lr_new = args.learning_rate * (0.1 ** (sum(epoch >= np.array(args.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
        # param_group['lr'] = opt.learning_rate