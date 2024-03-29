import random

import numpy as np
import torch

from Script.PredictModel import SeqBasedModel, PointBasedModel


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
    if args['model'] in ['LSTM', 'GRU', 'DIN', 'DIEN', 'Pointer']:
        model = SeqBasedModel(
            input_size=args['input_size'],
            embed_size=args['embed_size'],
            hidden_size=args['hidden_size'],
            predict_hidden_sizes=args['predict_hidden_sizes'],
            output_size=args['output_size'],
            dropout=args['dropout_rate'],
            model_name=args['model'],
            sparse_embedding=args['dataset'] in ('MQ2007',)
        )
    elif args['model'] in ['FM', 'DeepFM', 'PNN']:
        model = PointBasedModel(
            input_size=args['input_size'],
            embed_size=args['embed_size'],
            predict_hidden_sizes=args['predict_hidden_sizes'],
            output_size=args['output_size'],
            dropout=args['dropout_rate'],
            model_name=args['model'],
            sparse_embedding=args['dataset'] in ('MQ2007',)
        )
    else:
        raise NotImplementedError
    return model


class Evaluate:
    def __init__(self, criterion=None, evaluate_lens=None, mode=0):
        self.mode = mode
        self.criterion = criterion
        if evaluate_lens is None:
            evaluate_lens = [5, 10]
        self.evaluate_lens = np.array(evaluate_lens)
        self.loc = 1.0 / (np.arange(max(self.evaluate_lens)) + 1)
        self.loc = np.expand_dims(self.loc, 0)
        self.loc_log = 1 / np.log2(np.arange(max(self.evaluate_lens)) + 2)
        self.idcg = np.array([sum(self.loc_log[:_]) for _ in self.evaluate_lens])  # 理想情况全排对，即relevance全是1
        self.loc_log = np.expand_dims(self.loc_log, 0)
        self.evaluate_lens -= 1

    def evaluate(self, y_, y, with_loss=False, detail_acc=False):
        # loss = self.criterion(y_.view(y.numel(), -1).squeeze(), y.view(-1)) if with_loss else None
        loss = self.criterion(y_, y) if with_loss else None
        y_, y = [_.detach().cpu().numpy() for _ in [y_, y]]
        if self.mode == 0:  # this case is 0-1 label
            y_ = np.equal(np.argsort(-y_, axis=1), np.argsort(-y, axis=1))
        else:  # this case is directly sort label
            y_ = np.equal(np.argmax(y_, axis=-1), y)
        ndcgs = self.ndcg(y_)
        maps = self.map(y_)
        if detail_acc:
            return loss, ndcgs, maps, self.acc(y_)
        return loss, ndcgs, maps

    def ndcg(self, label):  # y_ is the relevance
        dcgs = np.cumsum(label * self.loc_log, axis=-1)[:, self.evaluate_lens]
        dcgs = np.mean(dcgs, axis=0)
        return dcgs / self.idcg

    def map(self, label):
        count = np.cumsum(label, axis=-1, dtype=float) + 1e-9
        maps = label * count
        maps = np.cumsum(maps * self.loc, axis=-1)[:, self.evaluate_lens]
        count = count[:, self.evaluate_lens]
        return np.mean(maps / count, axis=0)

    @staticmethod
    def acc(label):
        return np.mean(label, axis=0)

# def evaluate_utils_point(y_, y, criterion=None):
#     loss = None if criterion is None else criterion(y_, y)
#     y_, y = [_.detach().cpu().numpy() for _ in [y_, y]]
#     acc = np.mean(np.equal(y_ >= 0.5, y))
#     return loss, acc, roc_auc_score(y_true=y, y_score=y_)
#
#
# def evaluate_utils_seq(y_, y, criterion=None):
#     y_ = y_.view(-1, y_.size(-1))
#     y = y.view(-1)
#     loss = None if criterion is None else criterion(y_, y)
#     y_, y = [_.detach().cpu().numpy() for _ in [y_, y]]
#     acc = np.mean(np.equal(np.argmax(y_, axis=-1), y))
#     return loss, acc, roc_auc_score(y_true=y, y_score=y_, multi_class='ovr')
