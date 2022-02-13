import time

import numpy as np
import torch
from torch.utils import data


class SimpleDataset(data.Dataset):
    def __init__(self, user_item, mode='train', set_or_seq_len=5):
        super(SimpleDataset, self).__init__()
        self.user_log = user_item['log'][:, -4 * set_or_seq_len:]
        self.fields_num = user_item['fields']
        self.fields_num_sum = sum(self.fields_num)
        self.set_or_seq_len = set_or_seq_len
        self.item_loc_dict = {'train': 0, 'valid': 1, 'test': 2}
        self.item_loc = self.item_loc_dict[mode]

    def change_mode(self, mode='train'):
        self.item_loc = self.item_loc_dict[mode]

    def __len__(self):
        return self.user_log.shape[0]

    def __getitem__(self, user):
        logs = self.user_log[user][self.item_loc * self.set_or_seq_len:
                                   (self.item_loc + 2) * self.set_or_seq_len]
        x, y = logs[:, :-2], logs[self.set_or_seq_len:, -1]
        r = np.random.permutation(self.set_or_seq_len)
        return x, y, r


def collate_fn_point(data_):
    logs = torch.tensor(np.stack([_[0] for _ in data_]))
    labels = torch.tensor(np.stack([_[1] for _ in data_]), dtype=torch.float32)
    return logs, labels


def collate_fn_seq(data_):
    logs, labels = collate_fn_point(data_)
    random_seq = torch.tensor(np.stack([_[-1] for _ in data_]))
    labels = torch.argsort(-labels, dim=-1)
    labels = torch.gather(labels, 1, random_seq)
    logs[:, labels.size(1):] = torch.gather(logs[:, labels.size(1):], 1,
                                             random_seq.unsqueeze(-1).expand(-1, -1, logs.size(-1)))
    return logs, labels


if __name__ == '__main__':
    import os

    dataset = '../Data/alipay'
    train_data = SimpleDataset(np.load(os.path.join(dataset, 'user_item.npz')), mode='train')
    train_dataloader = data.DataLoader(train_data, batch_size=20,
                                       num_workers=4, shuffle=True)
    t1_ = t1 = time.perf_counter()
    network_time = 0
    print(train_data.fields_num)
    for i, (X, Y) in enumerate(train_dataloader):
        if i > 2:
            break
        network_time += time.perf_counter() - t1_
        print(X, X.shape)
        print(Y)
        t1_ = time.perf_counter()
    print(time.perf_counter() - t1, network_time)
