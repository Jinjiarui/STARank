import os
import pickle
import sys
import os
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
import random
import time
import pickle as pkl
import numpy as np
import datetime

from utils import  evaluate, get_aggregated_batch, construct_list, evaluate_multi, construct_list_with_profile, construct_list_with_profile_sim_hist
from model import MIR
# from click_models import DCM


def eval(model, sess, data, max_time_len, props, reg_lambda, batch_size, isrank, metric_scope):
    preds = []
    # labels = []
    losses = []
    fi_mat, ii_mat = [], []

    data_size = len(data[0])
    batch_num = data_size // batch_size
    print('eval', batch_size, batch_num)

    t = time.time()
    for batch_no in range(batch_num):
        data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
        pred, label, loss, fi, ii = model.eval(sess, data_batch, reg_lambda, no_print=batch_no)
        preds.extend(pred)
        # labels.extend(label)
        losses.append(loss)
        fi_mat.extend(fi)
        ii_mat.extend(ii)

    loss = sum(losses) / len(losses)
    cates = np.reshape(np.array(data[1])[:, :, 1], [-1, max_time_len]).tolist()
    hist_cates = np.reshape(np.array(data[3])[:, :, 1], [-1, max_behavior_len]).tolist()
    labels = data[5]
    # labels = data[-1]
    poss = data[-2]


    # if not os.path.exists('logs_{}/{}/'.format(data_set_name, max_time_len)):
    #     os.makedirs('logs_{}/{}/'.format(data_set_name, max_time_len))
    # model_name = '{}_{}_{}_{}_{}_{}'.format(initial_rankers, model_type, batch_size, lr, reg_lambda, hidden_size)
    #
    # if isrank:
    #     pickle.dump([preds, ], open('logs_{}//{}/{}_res.pkl'.format(data_set_name, max_time_len, model_name), 'wb'))
    # else:
    #     pickle.dump([preds, cates], open('logs_{}//{}/{}_init.pkl'.format(data_set_name, max_time_len, model_name), 'wb'))


    res = evaluate_multi(labels, preds, metric_scope, props, cates, poss, isrank)

    print("EVAL TIME: %.4fs" % (time.time() - t))
    print('fi mat', len(fi_mat))
    return loss, res, [fi_mat, ii_mat, cates, hist_cates, labels, preds]


def train(train_file, test_file, props, model_type, batch_size, feature_size, eb_dim, hidden_size, max_time_len,
          max_seq_len, itm_spar_fnum, itm_dens_fnum, hist_spar_fnum, hist_dens_fnum, profile_num, max_norm, metric_scope,
          intra_list, intra_set, set2list, loss, fi, ii, decay, epoch_num, keep_prob):
    tf.reset_default_graph()

    if model_type == 'MIR':
        model = MIR(feature_size, eb_dim, hidden_size, max_time_len, max_seq_len, itm_spar_fnum, itm_dens_fnum,
                    hist_spar_fnum, hist_dens_fnum, profile_num, max_norm, intra_list, intra_set, set2list, loss,
                    fi, ii, decay)
    else:
        print('No Such Model')
        exit()


    training_monitor = {
        'train_loss': [],
        'vali_loss': [],
        # 'map_l': [],
        # 'ndcg_l': [],
        # 'clicks_l': [],
        'utility_l': [],
        # 'de_ndcg_l': [],
    }

    model_name = '{}__{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(runner, date, initial_rankers, model_type, batch_size, lr,
               reg_lambda, hidden_size, embedding_size, intra_list, intra_set, set2list, loss, fi, ii)
    if not os.path.exists('logs/{}/{}/{}'.format(data_set_name, max_time_len, max_seq_len)):
        os.makedirs('logs/{}/{}/{}'.format(data_set_name, max_time_len, max_seq_len))
    if not os.path.exists('save_model/{}/{}/{}/'.format(data_set_name, max_time_len, model_name)):
        os.makedirs('save_model/{}/{}/{}/'.format(data_set_name, max_time_len, model_name))
    save_path = 'save_model/{}/{}/{}/ckpt'.format(data_set_name, max_time_len, model_name)
    log_save_path = 'logs/{}/{}/{}/{}.metrics'.format(data_set_name, max_time_len, max_seq_len, model_name)
    inter_log_path = 'logs_{}/{}.log'.format(data_set_name, model_name)

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_losses_step = []

        # before training process
        step = 0
        vali_loss, res, inter_log = eval(model, sess, test_file, max_time_len, props, reg_lambda, batch_size, False, metric_scope)

        training_monitor['train_loss'].append(None)
        training_monitor['vali_loss'].append(None)
        # training_monitor['map_l'].append(res[0][0])
        # training_monitor['ndcg_l'].append(res[1][0])
        # training_monitor['de_ndcg_l'].append(res[2][0])
        # training_monitor['clicks_l'].append(res[3][0])
        training_monitor['utility_l'].append(res[4][0])

        print("STEP %d  INTIAL RANKER | LOSS VALI: NULL" % step)
        for i, s in enumerate(metric_scope):
            print("@%d  MAP: %.4f  NDCG: %.4f  debiasNDCG: %.4f  CLICKS@L: %.4f  UTILITY@L: %.8f" % (
                  s, res[0][i], res[1][i], res[2][i], res[3][i], res[4][i]))

        early_stop = False

        data = train_file
        data_size = len(data[0])
        batch_num = data_size // batch_size
        eval_iter_num = (data_size // 5) // batch_size
        # eval_iter_num = (data_size // 30) // batch_size
        print('train', data_size, batch_num)

        # begin training process

        for epoch in range(epoch_num):
            # if early_stop:
            #     break
            for batch_no in range(batch_num):
                data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
                # if early_stop:
                #     break
                loss = model.train(sess, data_batch, lr, reg_lambda, keep_prob)
                step += 1
                train_losses_step.append(loss)

                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    training_monitor['train_loss'].append(train_loss)
                    train_losses_step = []

                    vali_loss, res, inter_log = eval(model, sess, test_file, max_time_len, props, reg_lambda, batch_size, True, metric_scope)
                    training_monitor['train_loss'].append(train_loss)
                    training_monitor['vali_loss'].append(vali_loss)
                    # training_monitor['map_l'].append(res[0][0])
                    # training_monitor['ndcg_l'].append(res[1][0])
                    # training_monitor['de_ndcg_l'].append(res[2][0])
                    # training_monitor['clicks_l'].append(res[3][0])
                    training_monitor['utility_l'].append(res[4][1])

                    print("EPOCH %d STEP %d  LOSS TRAIN: %.4f | LOSS VALI: %.4f" % (epoch, step, train_loss, vali_loss))
                    for i, s in enumerate(metric_scope):
                        print("@%d  MAP: %.4f  NDCG: %.4f  debiasNDCG: %.4f  CLICKS@L: %.4f  UTILITY@L: %.8f" % (
                            s, res[0][i], res[1][i], res[2][i], res[3][i], res[4][i]))

                    if training_monitor['utility_l'][-1] > max(training_monitor['utility_l'][:-1]):
                        # save model
                        model.save(sess, save_path)
                        pkl.dump(res[-1], open(log_save_path, 'wb'))
                        pkl.dump(inter_log, open(inter_log_path, 'wb'))
                        print('model saved')

                    if len(training_monitor['vali_loss']) > 2 and epoch > 0:
                        if (training_monitor['vali_loss'][-1] > training_monitor['vali_loss'][-2] and
                                training_monitor['vali_loss'][-2] > training_monitor['vali_loss'][-3]):
                            early_stop = True
                        if (training_monitor['vali_loss'][-2] - training_monitor['vali_loss'][-1]) <= 0.001 and (
                                training_monitor['vali_loss'][-3] - training_monitor['vali_loss'][-2]) <= 0.001:
                            early_stop = True



if __name__ == '__main__':
    # parameters
    random.seed(1234)
    runner = sys.argv[1]
    # runner = ''
    data_dir = 'data/'
    date = datetime.date.today().strftime('%Y_%m_%d')
    # data_set_name = 'prm'
    data_set_name = 'ad'
    if data_set_name == 'prm':
        processed_dir = os.path.join(data_dir, data_set_name + '/processed5/')
    else:
        processed_dir = os.path.join(data_dir, data_set_name + '/processed_5_init10/')

    stat_dir = os.path.join(processed_dir, 'data.stat')

    # initial ranker
    # initial_rankers = 'svm'
    # initial_rankers = 'mart'
    initial_rankers = 'DIN'

    model_type = 'MIR' #
    if data_set_name == 'prm':
        max_time_len = 30  # 30
    else:
        max_time_len = 10  # 10
    max_behavior_len = 30  # 30

    # training parameters
    reg_lambda = 5e-5  # default 1e-4
    lr = 1e-4 # default 1e-4
    keep_prob = 0.8  # default 0.8
    embedding_size = 16   # default 16
    batch_size = 16  # default 16
    hidden_size = 64  # default 64
    epoch_num = 100   # default 30
    max_norm = None  # default None

    if data_set_name == 'prm':
        metric_scope = [5, 10, 15, 20]
    else:
        metric_scope = [1, 3, 5, 10]

    # architecture
    intra_list = True  # True
    intra_set = True  # True
    fi = True  # True
    ii = True  # True
    decay = True # True
    set2list = 'SLAttention'  # SLAttention
    #set2list = 'GA'  # SLAttention
    loss = 'log'   # log

    stat = pkl.load(open(stat_dir, 'rb'))
    if data_set_name == 'prm':
        num_user, num_item, num_cate, num_ft, num_list, user_fnum, profile_fnum, itm_spar_fnum, itm_dens_fnum = \
            stat['user_num'], stat['item_num'], stat['cate_num'], stat['ft_num'], stat['list_num'], stat['user_fnum'], \
            stat['profile_fnum'], stat['itm_spar_fnum'], stat['itm_dens_fnum']
        hist_spar_fnum, hist_dens_fnum = itm_spar_fnum, itm_dens_fnum
    else:
        num_user, num_item, num_cate, num_ft, num_list, user_fnum, profile_fnum, itm_spar_fnum, itm_dens_fnum, \
        hist_spar_fnum, hist_dens_fnum = stat['user_num'], stat['item_num'], stat['cate_num'], stat['ft_num'], \
        stat['list_num'], stat['user_fnum'], stat['profile_fnum'], stat['itm_spar_fnum'], stat['itm_dens_fnum'], \
        stat['hist_spar_fnum'], stat['hist_dens_fnum']

    # user_profile_dict, train_file, val_file, test_file = pkl.load(open(os.path.join(processed_dir, 'data.data'), 'rb'))
    props = pkl.load(open(os.path.join(processed_dir, 'prop'), 'rb'))
    props[0] = [1e-6 for i in range(max_time_len)]
    profile = pkl.load(open(os.path.join(processed_dir, 'user.profile'), 'rb'))

    # construct training files
    if data_set_name == 'ad':
        train_dir = os.path.join(processed_dir, initial_rankers + '.data.train.time.' + str(max_behavior_len))
        use_pos = False
    else:
        train_dir = os.path.join(processed_dir, initial_rankers + '.data.train.' + str(max_behavior_len))
        use_pos = True
    if os.path.isfile(train_dir):
        train_lists = pkl.load(open(train_dir, 'rb'))
    else:
        train_lists = construct_list_with_profile(os.path.join(processed_dir, initial_rankers + '.rankings.train'), max_time_len,
                                     max_behavior_len, props, profile, use_pos)
        pkl.dump(train_lists, open(train_dir, 'wb'))

    # construct test files
    if data_set_name == 'ad':
        # test_dir = os.path.join(processed_dir, initial_rankers + '.data.test.time.sim.' + str(max_behavior_len))
        test_dir = os.path.join(processed_dir, initial_rankers + '.data.test.time.' + str(max_behavior_len))
        use_pos = False
    else:
        # test_dir = os.path.join(processed_dir, initial_rankers + '.data.test.sim.' + str(max_behavior_len))
        test_dir = os.path.join(processed_dir, initial_rankers + '.data.test.' + str(max_behavior_len))
        user_pos = True
    if os.path.isfile(test_dir):
        test_lists = pkl.load(open(test_dir, 'rb'))
    else:
        # test_lists = construct_list_with_profile_sim_hist(os.path.join(processed_dir, initial_rankers + '.rankings.test'), max_time_len,
        #                             max_behavior_len, props, profile, profile_fnum, use_pos)
        test_lists = construct_list_with_profile(os.path.join(processed_dir, initial_rankers + '.rankings.test'), max_time_len,
                                    max_behavior_len, props, profile, use_pos)
        pkl.dump(test_lists, open(test_dir, 'wb'))

    train(train_lists, test_lists, props, model_type, batch_size, num_ft, embedding_size, hidden_size, max_time_len,
          max_behavior_len, itm_spar_fnum, itm_dens_fnum, hist_spar_fnum, hist_dens_fnum, profile_fnum,  max_norm,
          metric_scope, intra_list, intra_set, set2list, loss, fi, ii, decay, epoch_num, keep_prob)




