import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm


def do_remap(old_array):
    array = np.asarray(old_array, dtype=np.int32)
    field_num = np.shape(array)[-1] - 2  # the last two is time and label
    fields_feature_num = np.zeros(field_num, dtype=np.int32)  # 记录每个特征域最大特征数目
    fields_feature_num[0] = np.max(array[:, 0]) + 1
    for field in range(1, field_num):
        remap_index = np.unique(array[:, field], return_inverse=True)[1]
        fields_feature_num[field] = np.max(remap_index) + 1
        array[:, field] = remap_index + np.sum(fields_feature_num[:field])
    return array, fields_feature_num


def get_sub_sequence(user_item, dataset_name, new_folder='./Data', max_len=20):
    old_folder = f'../STARec/Data/TaoBao/{dataset_name}/raw_data/'
    new_folder = os.path.join(new_folder, dataset_name)
    os.makedirs(new_folder, exist_ok=True)
    with np.load(os.path.join(old_folder, user_item)) as user_item:
        log, begin_len = user_item['log'], user_item['begin_len']
    every_last_index = []
    for (begin_loc, seq_len) in tqdm(begin_len):
        end_loc = begin_loc + seq_len
        every_last_index += list(range(end_loc - max_len, end_loc))
    log_user_last = log[every_last_index]
    log_user_last, fields_feature_num = do_remap(log_user_last)
    print(fields_feature_num)
    log_user_last = np.reshape(log_user_last, (-1, max_len, log_user_last.shape[-1]))
    print(fields_feature_num.dtype)
    np.savez(os.path.join(new_folder, 'user_item.npz'),
             log=log_user_last.astype(np.int32),
             fields=fields_feature_num)


def get_sliding_sequence(user_item, dataset_name, new_folder='./SlideData', sub_len=10):
    old_folder = f'../STARec/Data/TaoBao/{dataset_name}/raw_data/'
    new_folder = os.path.join(new_folder, dataset_name)
    os.makedirs(new_folder, exist_ok=True)
    with np.load(os.path.join(old_folder, user_item)) as user_item:
        log, begin_len = user_item['log'], user_item['begin_len']
    useful_index = []
    min_len = sub_len * 3 + sub_len
    train_i, valid_i, test_i = [], [], []
    n = 0
    for (begin_loc, seq_len) in tqdm(begin_len):
        if seq_len < min_len:  # For train, valid, test
            continue
        useful_index += list(range(begin_loc, begin_loc + seq_len - seq_len % 10))
        temp = list(range(n, n + seq_len // 10 - 1))
        train_i += temp[:-2]
        valid_i.append(temp[-2])
        test_i.append(temp[-1])
        n += seq_len // 10
    useful_log = log[useful_index]
    print("Log Len After filter:{}".format(len(useful_log)))
    log_user_last, fields_feature_num = do_remap(useful_log)
    print(log_user_last.dtype)
    log_save = {'log': log_user_last, 'fields': fields_feature_num,
                'train': train_i, 'test': test_i, 'valid': valid_i}
    np.savez(os.path.join(new_folder, 'user_item.npz'), **log_save)


def load_data(file_name):
    data_ = np.loadtxt(file_name, dtype=str, delimiter=' ', usecols=range(48))
    label = data_[:, 0:1].astype(np.float32)
    data_ = data_[:, 1:].reshape(-1)
    data_ = np.asarray([_[1] for _ in np.char.split(data_, ':')], dtype=np.float32)
    data_ = data_.reshape(-1, 47)
    return np.concatenate((data_, label), -1)


def dealMQ2007(new_folder='./SlideData', sub_len=10):
    old_folder = './Data/MQ2007-list'
    # old_folder = '../UnbiasedGraph/data/MQ2007'
    new_folder = os.path.join(new_folder, 'MQ2007')
    os.makedirs(new_folder, exist_ok=True)
    txt_file = os.path.join(old_folder, 'I{}.txt')

    with Pool(5) as pool:
        data = pool.map(load_data, [txt_file.format(i) for i in range(1, 6)])
    data = np.concatenate(data)
    data = np.split(data[:, 1:], np.where(np.diff(data[:, 0]))[0] + 1)
    min_len = sub_len * 3 + sub_len
    train_i, valid_i, test_i = [], [], []
    n = 0
    data = [_ for _ in data if len(_) > min_len]
    listwise_label = []
    for _ in tqdm(data):
        t = _[:, -1].copy()
        _[:, -1] = t > np.median(t)
        listwise_label.append(t)
        np.random.shuffle(_)
    listwise_label = np.concatenate(listwise_label)
    seqs_len = np.asarray([len(_) for _ in data])
    begin_len = np.concatenate(([0], np.cumsum(seqs_len)[:-1]))
    for (begin_loc, seq_len) in tqdm(zip(begin_len, seqs_len)):
        temp = list(range(n, n + seq_len // 10 - 1))
        train_i += temp[:-2]
        valid_i.append(temp[-2])
        test_i.append(temp[-1])
        n += seq_len // 10
    data = np.concatenate(data)
    print(len(train_i), len(test_i))
    print("Log Len After filter:{}".format(len(data)))
    print(data)
    log_save = {'log': data, 'fields': [46], 'listwise_label': listwise_label,
                'train': train_i, 'test': test_i, 'valid': valid_i}
    np.savez(os.path.join(new_folder, 'user_item.npz'), **log_save)


if __name__ == '__main__':
    # get_sliding_sequence('user_item.npz', 'taobao', sub_len=10)
    dealMQ2007()
