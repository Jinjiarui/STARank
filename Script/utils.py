import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from Script.PredictModel import SeqBasedModel


def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print("random seed set to be " + str(seed))


def load_model(args):
    if args['model'] in ['LSTM', 'GRU']:
        model = SeqBasedModel(
            input_size=args['input_size'],
            embed_size=args['embed_size'],
            hidden_size=args['hidden_size'],
            predict_hidden_sizes=args['predict_hidden_sizes'],
            output_size=args['output_size'],
            dropout_rate=args['dropout_rate'],
            model_name=args['model']
        )
    return model


def evaluate_utils(y_, y, criterion=None):
    loss = None if criterion is None else criterion(y_, y)
    acc = torch.mean(torch.eq(y_ > 0.5, y).float()).item()
    y_, y = [_.detach().cpu().numpy() for _ in [y_, y]]
    return loss, acc, roc_auc_score(y_true=y, y_score=y_)
