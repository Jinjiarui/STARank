import time

import numpy as np
import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, user_item, mode='train', seq_len=5, ratio=1.0, click_model=None):
        super(SimpleDataset, self).__init__()
        self.user_log = user_item['log'][:, -4 * seq_len:]
        if ratio < 1.0:
            select_random = np.random.choice(self.user_log.shape[0], int(self.user_log.shape[0] * ratio), replace=False)
            self.user_log = self.user_log[select_random]
        self.fields_num = user_item['fields']
        self.fields_num_sum = sum(self.fields_num)
        self.seq_len = seq_len
        self.item_loc_dict = {'train': 0, 'valid': 1, 'test': 2}
        self.item_loc = self.item_loc_dict[mode]

        labels = self.user_log[:, :, -1]
        labels = np.reshape(labels, (labels.shape[0], -1, seq_len))  # (B, 4, L)
        labels = labels[:, 1:]  # (B, 3, L)
        if not click_model is None:
            mask = 2 ** np.arange(seq_len).reshape((1, 1, seq_len))
            labels = click_model[np.sum(labels * mask, axis=-1)]
        self.labels = labels

    def change_mode(self, mode='train'):
        self.item_loc = self.item_loc_dict[mode]

    def __len__(self):
        return self.user_log.shape[0]

    def __getitem__(self, user):
        logs = self.user_log[user][self.item_loc * self.seq_len:
                                   (self.item_loc + 2) * self.seq_len]
        x, y = logs[:, :-2], self.labels[user][self.item_loc]
        return x, y


class SimpleDataset2(Dataset):
    def __init__(self, user_item, mode='train', seq_len=10, ratio=1.0,
                 click_model=None, delete_time=True, listwise_label=False):
        self.user_log = user_item['log']
        if delete_time:
            np.delete(self.user_log, -2, axis=1)
        # if listwise_label:
        #     self.user_log[:, -1] = user_item['listwise_label']
        self.item_loc_dict = {_: user_item[_] * seq_len for _ in ['train', 'valid', 'test']}
        self.item_loc = self.item_loc_dict[mode]
        if ratio < 1.0:
            select_random = np.random.choice(len(self.item_loc_dict['train']),
                                             int(len(self.item_loc_dict['train']) * ratio), replace=False)
            self.item_loc_dict['train'] = self.item_loc_dict['train'][select_random]
        if click_model is not None:
            mask = 2 ** np.arange(seq_len).reshape((1, 1, seq_len))
            labels = self.user_log[:, -1].reshape(-1, seq_len)
            labels = click_model[np.sum(labels * mask, axis=-1)].reshape(-1)
            self.user_log[:, -1] = labels
        self.fields_num = user_item['fields']
        self.fields_num_sum = sum(self.fields_num)
        self.seq_len = seq_len

    def change_mode(self, mode='train'):
        self.item_loc = self.item_loc_dict[mode]

    def __len__(self):
        return len(self.item_loc)

    def __getitem__(self, _):
        logs = self.user_log[self.item_loc[_]:self.item_loc[_] + 2 * self.seq_len]
        x, y = logs[:, :-1], logs[self.seq_len:, -1]
        return x, y


def collate_fn_point(data_):
    logs = torch.as_tensor(np.asarray([log for log, _ in data_]))
    labels = torch.as_tensor(np.asarray([label for _, label in data_], dtype=np.float32))
    random_seq = torch.argsort(torch.rand(labels.shape), -1)
    labels = torch.gather(labels, 1, random_seq)
    logs[:, labels.shape[1]:] = torch.gather(logs[:, labels.shape[1]:], 1,
                                             random_seq.unsqueeze(-1).expand(-1, -1, logs.shape[-1]))
    return logs, labels


def collate_fn_seq(data_):
    logs, labels = collate_fn_point(data_)
    labels = torch.argsort(-labels, dim=-1)
    return logs, labels


if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader

    dataset = '../SlideData/MQ2007'
    click_model_ = np.load('../SlideData/ClickModel/UBM.npy')
    train_data = SimpleDataset2(np.load(os.path.join(dataset, 'user_item.npz')), mode='train', delete_time=False)
    train_dataloader = DataLoader(train_data, batch_size=17, num_workers=4, shuffle=True, collate_fn=collate_fn_seq)
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
