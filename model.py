import itertools

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, static_bidirectional_rnn, LSTMCell, MultiRNNCell
import numpy as np
from tensorflow.python.framework import dtypes

from tensorflow.python.util import nest
import heapq


def tau_function(x):
    return tf.where(x > 0, tf.exp(x), tf.zeros_like(x))


def attention_score(x):
    return tau_function(x) / tf.add(tf.reduce_sum(tau_function(x), axis=1, keepdims=True), 1e-20)


class BaseModel(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, max_seq_len, itm_spar_num, itm_dens_num,
                 hist_spar_num, hist_dens_num, profile_num, max_norm=None):
        # reset graph
        tf.reset_default_graph()

        # input placeholders
        with tf.name_scope('inputs'):
            self.itm_spar_ph = tf.placeholder(tf.int32, [None, max_time_len, itm_spar_num], name='item_spar')
            self.itm_dens_ph = tf.placeholder(tf.float32, [None, max_time_len, itm_dens_num], name='item_dens')
            self.usr_profile = tf.placeholder(tf.int32, [None, profile_num], name='usr_profile')
            self.usr_spar_ph = tf.placeholder(tf.int32, [None, max_seq_len, hist_spar_num], name='user_spar')
            self.usr_dens_ph = tf.placeholder(tf.float32, [None, max_seq_len, hist_dens_num], name='user_dens')
            self.seq_length_ph = tf.placeholder(tf.int32, [None, ], name='seq_length_ph')
            self.hist_length_ph = tf.placeholder(tf.int32, [None, ], name='hist_length_ph')
            self.label_ph = tf.placeholder(tf.float32, [None, max_time_len], name='label_ph')
            self.time_ph = tf.placeholder(tf.float32, [None, max_seq_len], name='time_ph')
            self.is_train = tf.placeholder(tf.bool, [], name='is_train')


            # lr
            self.lr = tf.placeholder(tf.float32, [])
            # reg lambda
            self.reg_lambda = tf.placeholder(tf.float32, [])
            # keep prob
            self.keep_prob = tf.placeholder(tf.float32, [])
            self.max_time_len = max_time_len
            self.max_seq_len = max_seq_len
            self.hidden_size = hidden_size
            self.emb_dim = eb_dim
            self.itm_spar_num = itm_spar_num
            self.itm_dens_num = itm_dens_num
            self.hist_spar_num = hist_spar_num
            self.hist_dens_num = hist_dens_num
            self.profile_num = profile_num
            self.max_grad_norm = max_norm
            self.ft_num = itm_spar_num * eb_dim + itm_dens_num
            self.feature_size = feature_size

        # embedding
        with tf.name_scope('embedding'):
            self.emb_mtx = tf.get_variable('emb_mtx', [feature_size + 1, eb_dim],
                                           initializer=tf.truncated_normal_initializer)
            self.itm_spar_emb = tf.gather(self.emb_mtx, self.itm_spar_ph)
            self.usr_spar_emb = tf.gather(self.emb_mtx, self.usr_spar_ph)
            self.usr_prof_emb = tf.gather(self.emb_mtx, self.usr_profile)

            self.item_seq = tf.concat(
                [tf.reshape(self.itm_spar_emb, [-1, max_time_len, itm_spar_num * eb_dim]), self.itm_dens_ph], axis=-1)

    def build_fc_net(self, inp, scope='fc'):
        with tf.variable_scope(scope):
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=False)
            fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
            dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
            fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
            dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
            fc3 = tf.layers.dense(dp2, 2, activation=None, name='fc3')
            score = tf.nn.softmax(fc3)
            score = tf.reshape(score[:, :, 0], [-1, self.max_time_len])
            # output
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred

    def build_mlp_net(self, inp, layer=(500, 200, 80), scope='mlp'):
        with tf.variable_scope(scope):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn', training=False)
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
            final = tf.layers.dense(inp, 2, activation=None, name='fc_final')
            score = tf.nn.softmax(final)
            score = tf.reshape(score[:, :, 0], [-1, self.max_time_len])
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred

    def build_logloss(self, y_pred):
        # loss
        self.loss = tf.losses.log_loss(self.label_ph, y_pred)
        self.opt()

    def build_mseloss(self, y_pred):
        self.loss = tf.losses.mean_squared_error(self.label_ph, y_pred)
        self.opt()

    def build_attention_loss(self, y_pred):
        self.label_wt = attention_score(self.label_ph)
        self.pred_wt = attention_score(y_pred)
        # self.pred_wt = y_pred
        self.loss = tf.losses.log_loss(self.label_wt, self.pred_wt)
        # self.loss = tf.losses.mean_squared_error(self.label_wt, self.pred_wt)
        self.opt()

    def opt(self):
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
                # self.loss += self.reg_lambda * tf.norm(v, ord=1)

        # self.lr = tf.train.exponential_decay(
        #     self.lr_start, self.global_step, self.lr_decay_step,
        #     self.lr_decay_rate, staircase=True, name="learning_rate")

        self.optimizer = tf.train.AdamOptimizer(self.lr)

        if self.max_grad_norm is not None:
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
            self.train_step = self.optimizer.apply_gradients(grads_and_vars)
        else:
            self.train_step = self.optimizer.minimize(self.loss)

    def multihead_attention(self,
                            queries,
                            keys,
                            num_units=None,
                            num_heads=2,
                            scope="multihead_attention",
                            reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            outputs = tf.nn.dropout(outputs, self.keep_prob)
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        return outputs

    def bilstm(self, inp, hidden_size, scope='bilstm', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_fw')
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_bw')

            outputs, state_fw, state_bw = static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inp, dtype='float32')
        return outputs, state_fw, state_bw


    def train(self, sess, batch_data, lr, reg_lambda, keep_prob=0.8):
        de_lb = np.array(batch_data[-1])
        de_lb[de_lb > 10] = 10
        batch_data[-1] = de_lb
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={
            self.usr_profile: batch_data[0],
            self.itm_spar_ph: batch_data[1],
            self.itm_dens_ph: batch_data[2],
            self.usr_spar_ph: batch_data[3],
            self.usr_dens_ph: batch_data[4],
            # self.label_ph: batch_data[-1],
            self.label_ph: batch_data[5],
            self.time_ph: batch_data[6],
            self.seq_length_ph: batch_data[7],
            self.hist_length_ph: batch_data[8],
            self.lr: lr,
            self.reg_lambda: reg_lambda,
            self.keep_prob: keep_prob,
            self.is_train: True,
        })
        return loss

    def eval(self, sess, batch_data, reg_lambda, keep_prob=1, no_print=True):
        # fi_mat, ii_mat = [], []
        pred, label, loss, fi_mat, ii_mat = sess.run([self.y_pred, self.label_ph, self.loss, self.fi_mat, self.ii_mat], feed_dict={
        # pred, label, loss, fi_mat, ii_mat = sess.run([self.y_pred, self.label_ph, self.loss, self.a_v, self.a_q], feed_dict={
        # pred, label, loss = sess.run([self.y_pred, self.label_ph, self.loss], feed_dict={
            self.usr_profile: batch_data[0],
            self.itm_spar_ph: batch_data[1],
            self.itm_dens_ph: batch_data[2],
            self.usr_spar_ph: batch_data[3],
            self.usr_dens_ph: batch_data[4],
            # self.label_ph: batch_data[-1],
            self.label_ph: batch_data[5],
            self.time_ph: batch_data[6],
            self.seq_length_ph: batch_data[7],
            self.hist_length_ph: batch_data[8],
            self.reg_lambda: reg_lambda,
            self.keep_prob: keep_prob,
            self.is_train: False,
        })
        return pred.reshape([-1, self.max_time_len]).tolist(), label.reshape([-1, self.max_time_len]).tolist(), loss, fi_mat, ii_mat

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)


class deepFM(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, max_seq_len, itm_spar_num, itm_dens_num, hist_spar_num, hist_dens_num,
                 profile_num, max_norm=None, keep_prob_fm=[1.0, 1.0], deep_layer=[256, 128]):
        super(deepFM, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                  max_seq_len, itm_spar_num, itm_dens_num, hist_spar_num, hist_dens_num, profile_num, max_norm)

        with tf.variable_scope('deepFM'):

            self.feature_bias = tf.get_variable('feature_bias', [feature_size + 1, 1],
                                           initializer=tf.truncated_normal_initializer)

            # FM component
            first_order = tf.reduce_sum(tf.gather(self.feature_bias, self.itm_spar_ph), -1)
            first_order = tf.nn.dropout(first_order, keep_prob_fm[0])

            sum_square = tf.square(tf.reduce_sum(self.itm_spar_emb, 2))
            square_sum = tf.reduce_sum(tf.square(self.itm_spar_emb), 2)
            second_order = 0.5 * tf.subtract(sum_square, square_sum)
            second_order = tf.nn.dropout(second_order, keep_prob_fm[1])

            # deep component
            inp = self.item_seq
            for i, hidden_num in enumerate(deep_layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
            deep = inp

            final = tf.concat([first_order, second_order, deep], axis=-1)
            # final = deep
            self.y_pred = self.build_mlp_net(final)
            self.build_logloss(self.y_pred)


class MIR(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, max_seq_len, itm_spar_num, itm_dens_num,
                 hist_spar_num, hist_dens_num, profile_num, max_norm=None, intra_list=True, intra_set=True,
                 set2list='SLAttention', loss='log', fi=True, ii=True, decay=True):
        super(MIR, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                  max_seq_len, itm_spar_num, itm_dens_num, hist_spar_num, hist_dens_num, profile_num, max_norm)

        with tf.variable_scope('MIR'):
            self.istrain = tf.placeholder(tf.bool, [])
            self.item_mask = tf.expand_dims(tf.sequence_mask(self.seq_length_ph, maxlen=max_time_len, dtype=tf.float32),
                                  axis=-1)

            # intra-set interaction
            if intra_set:
                with tf.variable_scope('cross_item'):
                    item_seq = self.item_seq
                    seq = self.multihead_attention(item_seq, item_seq, num_heads=1)
                    seq = tf.concat([seq, self.item_seq], axis=-1)
                    self.cross_item_embed = seq * self.item_mask
            else:
                self.cross_item_embed = self.item_seq


            self.user_seq = tf.concat([tf.reshape(self.usr_spar_emb, [-1, max_seq_len, hist_spar_num * eb_dim]),
                                       self.usr_dens_ph], axis=-1)
            # intra-list interaction
            if intra_list:
                outputs, _, _ = self.bilstm(tf.unstack(self.user_seq, max_seq_len, 1), hidden_size, scope='user_bilstm')
                seq_ht = tf.reshape(tf.stack(outputs, axis=1), (-1, max_seq_len, hidden_size * 2))
                usr_seq = tf.concat([seq_ht, self.user_seq], -1)
            else:
                usr_seq = self.user_seq

            # set2list interaction
            with tf.variable_scope('set2list'):
                if set2list == 'co-att':
                    usr_seq = self.user_seq
                    v, q = self.co_attention(self.cross_item_embed, usr_seq)
                    seq = tf.concat([v, q], axis=-1)
                    self.set2list_embed = seq * self.item_mask
                elif set2list == 'SLAttention':
                    v, q = self.SLAttention(self.cross_item_embed, usr_seq, self.itm_spar_emb, self.usr_spar_emb,
                                            fi, ii, decay)
                    seq = tf.concat([v, q], axis=-1)
                    self.set2list_embed = seq * self.item_mask
                elif set2list == 'GA':
                    # usr_seq = self.multihead_attention(seq_ht, self.cross_item_embed, num_heads=1, scope='usr_att')
                    # item guided attention
                    itm_seq = self.multihead_attention(self.cross_item_embed, seq_ht, num_heads=1, scope='itm_att')
                    # seq = tf.concat([usr_seq, itm_seq], axis=-1)
                    seq = itm_seq
                    # seq = usr_seq
                    self.set2list_embed = seq * self.item_mask
                else:
                    self.set2list_embed = self.user_seq

            # mlp
            # self.final_embed = tf.concat([self.item_seq, self.intra_item_embed, self.set2list_embed], axis=-1)
            self.final_embed = tf.concat([self.item_seq, self.set2list_embed], axis=-1)
            # self.final_embed = self.set2list_embed
            self.y_pred = self.build_mlp_net(self.final_embed)

            # loss
            if loss == 'list':
                self.build_attention_loss(self.y_pred)
            elif loss == 'mse':
                self.build_mseloss(self.y_pred)
            else:
                self.build_logloss(self.y_pred)

    def feed_forward_net(self, inp, d_ff=256, scope='ffn'):
        with tf.variable_scope(scope):
            d_ft = inp.get_shape()[-1]
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)
            tmp = bn1 + inp
            fc1 = tf.layers.dense(tmp, d_ff, activation=tf.nn.relu, name='fc1')
            dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
            fc2 = tf.layers.dense(dp1, d_ft, activation=None, name='fc2')
            dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
            bn2 = tf.layers.batch_normalization(inputs=dp2, name='bn2', training=self.is_train)
        return inp + bn2

    def co_attention(self, V, Q, scope='co_att'):
        with tf.variable_scope(scope):
            v_dim, q_dim = V.get_shape()[-1], Q.get_shape()[-1]
            v_seq_len, q_seq_len = V.get_shape()[-2], Q.get_shape()[-2]
            bat_size = tf.shape(Q)[0]
            w_b = tf.get_variable("w_b", [1, q_dim, v_dim], initializer=tf.truncated_normal_initializer)
            C = tf.matmul(Q, tf.matmul(tf.tile(w_b, [bat_size, 1, 1]), tf.transpose(V, perm=[0, 2, 1])))
            C = tf.tanh(C)

            w_v = tf.get_variable('w_v', [v_dim, v_seq_len], initializer=tf.truncated_normal_initializer)
            w_q = tf.get_variable('w_q', [q_dim, v_seq_len], initializer=tf.truncated_normal_initializer)
            hv_1 = tf.reshape(tf.matmul(tf.reshape(V, [-1, v_dim]), w_v), [-1, v_seq_len, v_seq_len])
            hq_1 = tf.reshape(tf.matmul(tf.reshape(Q, [-1, q_dim]), w_q), [-1, q_seq_len, v_seq_len])
            hq_1 = tf.transpose(hq_1, perm=[0, 2, 1])
            h_v = tf.nn.tanh(hv_1 + tf.matmul(hq_1, C))
            h_q = tf.nn.tanh(hq_1 + tf.matmul(hv_1, tf.transpose(C, perm=[0, 2, 1])))
            a_v = tf.nn.softmax(h_v, axis=-1)
            a_q = tf.nn.softmax(h_q, axis=-1)
            self.a_v = a_v
            self.a_q = a_q
            v = tf.matmul(a_v, V)
            q = tf.matmul(a_q, Q)
        return v, q

    def SLAttention(self, V, Q, V_s, Q_s, fi=True, ii=True, decay=True, scope='fi_s2l'):
        with tf.variable_scope(scope):
            v_dim, q_dim = V.get_shape()[-1], Q.get_shape()[-1]
            v_seq_len, q_seq_len = V.get_shape()[-2], Q.get_shape()[-2]
            bat_size = tf.shape(Q)[0]

            # get affinity matrix
            if fi:
                self.w_b_fi = tf.get_variable("w_b_fi", [1, self.emb_dim, self.emb_dim],
                                              initializer=tf.truncated_normal_initializer)

                V_s = tf.reshape(V_s, [-1, self.max_time_len * self.itm_spar_num, self.emb_dim])
                Q_s = tf.reshape(Q_s, [-1, self.max_seq_len * self.hist_spar_num, self.emb_dim])
                C2 = tf.matmul(Q_s, tf.matmul(tf.tile(self.w_b_fi, [bat_size, 1, 1]), tf.transpose(V_s, perm=[0, 2, 1])))
                C2 = tf.layers.conv2d(tf.expand_dims(C2, -1), 1, self.hist_spar_num, strides=(self.hist_spar_num, self.itm_spar_num))
                C2 = tf.reshape(C2, [bat_size, q_seq_len, v_seq_len])
                self.fi_mat = C2
            if ii:
                self.w_b = tf.get_variable("w_b", [1, q_dim, v_dim], initializer=tf.truncated_normal_initializer)
                C1 = tf.matmul(Q, tf.matmul(tf.tile(self.w_b, [bat_size, 1, 1]), tf.transpose(V, perm=[0, 2, 1])))
                self.ii_mat = C1
                if fi:
                    C1 = C1 + C2
            else:
                C1 = C2

            if decay:
                # decay
                pos = tf.reshape(tf.tile(tf.expand_dims(self.time_ph, -1), [1, 1, v_seq_len]),
                                 [-1, q_seq_len, v_seq_len])
                usr_prof = tf.reshape(self.usr_prof_emb, [-1, self.profile_num * self.emb_dim])
                usr_prof = tf.layers.dense(usr_prof, 32, activation=tf.nn.relu, name='fc_decay1')
                theta = tf.layers.dense(usr_prof, 1, activation=tf.nn.relu, name='fc_decay2')
                self.decay_theta = tf.tile(tf.reshape(theta, [-1, 1, 1]), [1, q_seq_len, v_seq_len])
                pos_decay = tf.exp(self.decay_theta * pos)
                C = tf.tanh(C1 * pos_decay + C1)
            else:
                C = C1

            # attention map
            w_v = tf.get_variable('w_v', [v_dim, v_seq_len], initializer=tf.truncated_normal_initializer)
            w_q = tf.get_variable('w_q', [q_dim, v_seq_len], initializer=tf.truncated_normal_initializer)
            hv_1 = tf.reshape(tf.matmul(tf.reshape(V, [-1, v_dim]), w_v), [-1, v_seq_len, v_seq_len])
            hq_1 = tf.reshape(tf.matmul(tf.reshape(Q, [-1, q_dim]), w_q), [-1, q_seq_len, v_seq_len])
            hq_1 = tf.transpose(hq_1, perm=[0, 2, 1])
            h_v = tf.nn.tanh(hv_1 + tf.matmul(hq_1, C))
            # h_q = tf.nn.tanh(hq_1 + tf.matmul(hv_1, tf.transpose(C, perm=[0, 2, 1])))
            # h_v = tf.nn.tanh(tf.matmul(hq_1, C))
            h_q = tf.nn.tanh(tf.matmul(hv_1, tf.transpose(C, perm=[0, 2, 1])))
            a_v = tf.nn.softmax(h_v, axis=-1)
            a_q = tf.nn.softmax(h_q, axis=-1)
            self.a_v = a_v
            self.a_q = a_q
            v = tf.matmul(a_v, V)
            q = tf.matmul(a_q, Q)
        return v, q


class GSF(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, max_seq_len, itm_spar_num, itm_dens_num, hist_spar_num, hist_dens_num,
                 profile_num, max_norm=None, group_size=1, activation='relu', hidden_layer_size=[200, 80, 20]):
        super(GSF, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                  max_seq_len, itm_spar_num, itm_dens_num, hist_spar_num, hist_dens_num, profile_num, max_norm)
        self.group_size = group_size
        input_list = tf.unstack(self.item_seq, axis=1)
        input_data = tf.concat(input_list, axis=0)
        output_data = input_data
        if activation == 'elu':
            activation = tf.nn.elu
        else:
            activation = tf.nn.relu

        input_data_list = tf.split(output_data, self.max_time_len, axis=0)
        output_sizes = hidden_layer_size + [group_size]
        #
        output_data_list = [0 for _ in range(max_time_len)]
        group_list = []
        self.get_possible_group([], group_list)
        for group in group_list:
            group_input = tf.concat([input_data_list[idx]
                                     for idx in group], axis=1)
            group_score_list = self.build_gsf_fc_function(group_input, output_sizes, activation)
            for i in range(group_size):
                output_data_list[group[i]] += group_score_list[i]
        self.y_pred = tf.concat(output_data_list, axis=1)
        # self.y_pred = self.build_gsf_fc_function(self.item_seq, output_sizes, activation)
        # self.y_pred = tf.reshape(self.y_pred, [-1, self.max_time_len])
        self.build_logloss(self.y_pred)

    def build_gsf_fc_function(self, inp, hidden_size, activation, scope="gsf_nn"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for j in range(len(hidden_size)):
                bn = tf.layers.batch_normalization(inputs=inp, name='bn'+str(j), training=self.is_train)
                if j != len(hidden_size) - 1:
                    inp = tf.layers.dense(bn, hidden_size[j], activation=activation, name='fc' + str(j))
                else:
                    inp = tf.layers.dense(bn, hidden_size[j], activation=tf.nn.sigmoid, name='fc' + str(j))
        return tf.split(inp, self.group_size, axis=1)

    def get_possible_group(self, group, group_list):
        if len(group) == self.group_size:
            group_list.append(group)
            return
        else:
            for i in range(self.max_time_len):
                self.get_possible_group(group + [i], group_list)


class miDNN(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, max_seq_len, itm_spar_num, itm_dens_num,
                 hist_spar_num, hist_dens_num, profile_num, max_norm=None, hidden_layer_size=[200, 80, 20]):
        super(miDNN, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                           max_seq_len, itm_spar_num, itm_dens_num, hist_spar_num, hist_dens_num,
                                           profile_num, max_norm)
        fmax = tf.reduce_max(tf.reshape(self.item_seq, [-1, self.ft_num]), axis=0)
        fmin = tf.reduce_min(tf.reshape(self.item_seq, [-1, self.ft_num]), axis=0)
        global_seq = (self.item_seq - fmin)/(fmax - fmin)
        inp = tf.concat([self.item_seq, global_seq], axis=-1)

        self.y_pred = self.build_miDNN_net(inp, hidden_layer_size)
        self.build_logloss(self.y_pred)


    def build_miDNN_net(self, inp, layer, scope='mlp'):
        with tf.variable_scope(scope):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn', training=self.is_train)
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
            final = tf.layers.dense(inp, 1, activation=tf.nn.sigmoid, name='fc_final')
            score = tf.reshape(final, [-1, self.max_time_len])
        return score


class GlobalRerank(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, max_seq_len, itm_spar_num, itm_dens_num, hist_spar_num, hist_dens_num,
                 profile_num, max_norm=None, hidden_layer_size=[200, 80, 20], K=3):
        super(GlobalRerank, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                  max_seq_len, itm_spar_num, itm_dens_num, hist_spar_num, hist_dens_num, profile_num, max_norm)
        self.cell = GRUCell(hidden_size)
        with tf.name_scope('gru'):

            seq_ht, seq_final_state = tf.nn.dynamic_rnn(self.cell, inputs=self.item_seq,
                                                        sequence_length=self.seq_length_ph, dtype=tf.float32,
                                                        scope='gru1')
        seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
        self.w = tf.get_variable('h_w', [hidden_size, 1], initializer=tf.truncated_normal_initializer)
        y_pred = seq_mask * tf.reshape(tf.nn.sigmoid(tf.matmul(tf.reshape(seq_ht, [-1, hidden_size]), self.w)), [-1, self.max_time_len])
        # self.y_pred = self.build_GR_net(seq_ht, hidden_layer_size) * seq_mask
        self.build_attention_loss(y_pred)

        # batch_size = self.item_seq.get_shape()[0]
        list_len = tf.gather(self.seq_length_ph, 0)
        idx = tf.range(self.max_time_len)
        ft_list = tf.gather(self.item_seq, 0)
        a = ft_list.get_shape()
        a_print = tf.Print(a, ['ft list', a])
        beam = {tf.constant(0.0): ([], self.cell.zero_state(1, tf.float32))}
        for j in range(max_time_len):
            stepbeam = {}

            for c in beam.keys():
                candidate = {}
                prev_list, prev_h = beam[c]
                if j == 0:
                    remain = idx
                else:
                    remain = tf.sets.set_difference(idx, tf.cast(tf.stack(prev_list, axis=0), tf.int32))
                for i in range(max_time_len-len(prev_list)):
                    itm = remain[i]
                    # itm_print = tf.Print(itm, ['itm', itm])
                    # b = ft_list[itm].get_shape()
                    # b_print = tf.Print(b, ['b', b])
                    cur_h, state = self.cell(tf.reshape(tf.gather(ft_list, itm), [-1, self.ft_num]), prev_h)
                    score = tf.squeeze(tf.nn.sigmoid(cur_h * self.w))
                    cur_list = prev_list + [itm]
                    cur_c = c + score
                    candidate[cur_c] = (cur_list, cur_h)
                stepbeam.update(candidate)
                stepbeam = dict(heapq.nlargest(K, stepbeam.items(), key=lambda x: x[0]))
            beam = stepbeam
        c, v = heapq.nlargest(1, beam.items(), key=lambda x: x[0])
        self.y_pred = tf.concat([v[0][k] for k in range(max_time_len)], axis=-1)
        # idx = sorted(list(range(list_len)), key=lambda k: v[0][k])
        # self.y_pred = tf.concat(idx, axis=-1)





    def build_GR_net(self, inp, layer, scope='mlp'):
        with tf.variable_scope(scope):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn')
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
            final = tf.layers.dense(inp, 1, activation=tf.nn.sigmoid, name='fc_final')
            score = tf.reshape(final, [-1, self.max_time_len])
            return score



class PRM(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, max_seq_len, itm_spar_num, itm_dens_num, hist_spar_num, hist_dens_num,
                 profile_num, attention=False, max_norm=None, d_model=64, d_inner_hid=128, dropout=0.8):
        super(PRM, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                  max_seq_len, itm_spar_num, itm_dens_num, hist_spar_num, hist_dens_num, profile_num, max_norm)

        pos_dim = self.item_seq.get_shape().as_list()[-1]
        self.d_model = d_model
        if not attention:
            self.pos_mtx = tf.get_variable("pos_mtx", [max_time_len, pos_dim],
                                       initializer=tf.truncated_normal_initializer)
            self.item_seq = self.item_seq + self.pos_mtx
            if pos_dim % 2:
                self.item_seq = tf.pad(self.item_seq, [[0, 0], [0, 0], [0, 1]])
            # self.item_seq = tf.layers.dense(self.item_seq, self.d_model, activation=tf.nn.relu, name='fc')
            self.item_seq = self.multihead_attention(self.item_seq, self.item_seq)
            # self.item_seq = self.positionwise_feed_forward(self.item_seq, self.d_model, d_inner_hid, dropout)
        else:
            if pos_dim % 2:
                self.item_seq = tf.pad(self.item_seq, [[0, 0], [0, 0], [0, 1]])
            # self.item_seq = tf.layers.dense(self.item_seq, self.d_model, activation=tf.nn.relu, name='fc')
            self.item_seq = self.multihead_attention(self.item_seq, self.item_seq)

        mask = tf.expand_dims(tf.sequence_mask(self.seq_length_ph, maxlen=max_time_len, dtype=tf.float32), axis=-1)
        seq_rep = self.item_seq * mask

        if attention:
            self.y_pred = self.build_fc_net(seq_rep)
            self.build_attention_loss(self.y_pred)
        else:
            # self.build_prm_fc_function(seq_rep)
            self.y_pred = self.build_fc_net(seq_rep)
            self.build_logloss(self.y_pred)

    def positionwise_feed_forward(self, inp, d_hid, d_inner_hid, dropout=0.9):
        with tf.variable_scope('pos_ff'):
            l1 = tf.layers.conv1d(inp, d_inner_hid, 1, activation='relu')
            l2 = tf.layers.conv1d(l1, d_hid, 1)
            dp = tf.nn.dropout(l2, dropout, name='dp')
            dp = dp + inp
            output = tf.layers.batch_normalization(inputs=dp, name='bn', training=self.is_train)
        return output

    def build_prm_fc_function(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
        fc3 = tf.layers.dense(dp2, 1, activation=None, name='fc3')
        score = tf.nn.softmax(tf.reshape(fc3, [-1, self.max_time_len]))
        # output
        seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
        return seq_mask * score


class DLCM(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, max_seq_len, itm_spar_num, itm_dens_num, hist_spar_num, hist_dens_num,
                 profile_num, max_norm=None):
        super(DLCM, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                   max_seq_len, itm_spar_num, itm_dens_num, hist_spar_num, hist_dens_num,  profile_num, max_norm)
        with tf.name_scope('gru'):
            seq_ht, seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.item_seq,
                                                        sequence_length=self.seq_length_ph, dtype=tf.float32,
                                                        scope='gru1')
        self.y_pred = self.build_phi_function(seq_ht, seq_final_state, hidden_size)
        self.build_attention_loss(self.y_pred)
        # self.build_logloss()

    def build_phi_function(self, seq_ht, seq_final_state, hidden_size):
        bn1 = tf.layers.batch_normalization(inputs=seq_final_state, name='bn1', training=self.is_train)
        seq_final_fc = tf.layers.dense(bn1, hidden_size, activation=tf.nn.tanh, name='fc1')
        dp1 = tf.nn.dropout(seq_final_fc, self.keep_prob, name='dp1')
        seq_final_fc = tf.expand_dims(dp1, axis=1)
        bn2 = tf.layers.batch_normalization(inputs=seq_ht, name='bn2', training=self.is_train)
        fc2 = tf.layers.dense(tf.multiply(bn2, seq_final_fc), 2, activation=None, name='fc2')
        score = tf.nn.softmax(fc2)
        score = tf.reshape(score[:, :, 0], [-1, self.max_time_len])
        # sequence mask
        seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
        return score * seq_mask


class Seq2Seq(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, max_seq_len, itm_spar_num, itm_dens_num,hist_spar_num, hist_dens_num,
                 profile_num, max_norm=None, num_layers=1, num_glimpse=1, init_min_val=-0.08, init_max_val=0.08):
        super(Seq2Seq, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                      max_seq_len, itm_spar_num, itm_dens_num, hist_spar_num, hist_dens_num, profile_num, max_norm)
        self.global_step = tf.Variable(0, trainable=False)
        self.input_dim = self.ft_num
        self.embeded_enc_inputs = self.item_seq
        self.num_layers = num_layers
        self.num_glimpse = num_glimpse
        self.initializer = tf.random_uniform_initializer(init_min_val, init_max_val)
        self.target_seq = tf.contrib.framework.argsort(self.label_ph, axis=-1, direction='DESCENDING')
        self.attention_dim = hidden_size

        self.enc_seq_length = self.seq_length_ph
        self.target_seq_length = self.seq_length_ph
        self.batch_size = tf.shape(self.embeded_enc_inputs)[0]

        with tf.variable_scope("encoder"):
            self.enc_cell = LSTMCell(self.hidden_size, initializer=self.initializer)

            if self.num_layers > 1:
                cells = [self.enc_cell] * self.num_layers
                self.enc_cell = MultiRNNCell(cells)

            # self.encoder_outputs : [None, max_time, output_size]
            self.enc_outputs, self.enc_final_states = tf.nn.dynamic_rnn(
                self.enc_cell, self.embeded_enc_inputs,
                self.enc_seq_length, dtype=tf.float32)

        with tf.variable_scope("decoder"):
            # [[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]],
            #                        [[0, 2], [1, 3], [2, 1]]]
            self.dec_cell = LSTMCell(self.hidden_size, initializer=self.initializer)

            if self.num_layers > 1:
                cells = [self.dec_cell] * self.num_layers
                self.dec_cell = MultiRNNCell(cells)

            self.dec_outputs, self.dec_final_states = tf.nn.dynamic_rnn(
                self.dec_cell, self.embeded_enc_inputs,
                self.enc_seq_length, self.enc_final_states)
            self.y_pred = self.build_fc_net(self.dec_outputs)

        self.loss = tf.losses.mean_squared_error(self.target_seq, -self.y_pred)
        # self.loss = tf.losses.log_loss(self.label_ph, self.y_pred)
        self.opt()

