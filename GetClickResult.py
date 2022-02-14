import numpy as np
import os
from Script.Click import PositionBiasedModel, UserBrowsingModel, ClickClainModel


class GetClickResult:
    def __init__(self, seq_len, data_dir='./Data'):
        self.seq_len = seq_len
        self.data_dir = os.path.join(data_dir, 'ClickModel')
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.possible_orders = self.get_possible_order()
        self.relevance_level = 2

    def get_possible_order(self):
        possible_orders = np.arange(2 ** self.seq_len)
        mask = 2 ** np.arange(self.seq_len)
        possible_orders = np.bitwise_and(np.expand_dims(possible_orders, -1), mask)
        possible_orders[possible_orders > 0] = 1
        return possible_orders[:, ::-1]

    def generate_PBM(self):
        click_model = PositionBiasedModel()
        click_model.set_observe_probability()
        clicks, observes = [], []
        for order in self.possible_orders:
            iter_observe_list = []
            iter_click_list = []
            for _p, _r in enumerate(order):
                iter_observe, iter_click = click_model.get_click_probability(relevance=_r,
                                                                             relevance_level=self.relevance_level,
                                                                             position=_p)
                iter_observe_list.append(iter_observe)
                iter_click_list.append(iter_click)
            observes.append(iter_observe_list)
            clicks.append(iter_click_list)
        observes, clicks = [np.array(_) for _ in (observes, clicks)]
        self.save_matrix(np.random.binomial(1, clicks), 'PBM')
        print(observes)
        print(clicks)

    def generate_CCM(self, non_observe=0.4, click_observe=0.27):
        clicks, observes = [], []
        click_model = ClickClainModel(non_observe, click_observe)
        counter, observes_pro, click_pro = 0, 0, 0
        for order in self.possible_orders:
            iter_observe_list, iter_click_list = [], []
            iter_observe, iter_click = 1, 0
            for _p, iter_relevance in enumerate(order):
                iter_observe_pro, iter_click_pro = click_model.set_browse_probability(iter_observe, iter_click,
                                                                                      iter_relevance,
                                                                                      relevance_level=self.relevance_level)
                iter_observe = 1 if np.random.random_sample(1) <= iter_observe_pro else 0
                iter_click = 1 if np.random.random_sample(1) <= iter_click_pro else 0
                iter_observe_list.append(iter_observe)
                iter_click_list.append(iter_click)
                counter += 1
                observes_pro += iter_observe
                click_pro += iter_click
            clicks.append(iter_click_list)
            observes.append(iter_observe_list)
        observes, clicks = [np.array(_) for _ in (observes, clicks)]
        self.save_matrix(clicks, 'CCM')
        print(observes)
        print(clicks)

    def generate_UBM(self):
        clicks, observes = [], []
        click_model = UserBrowsingModel()
        click_model.set_observe_probability()
        for _r in self.possible_orders:
            iter_click_list, exam_p_list = click_model.sampleClicksForOneList(_r, relevance_level=self.relevance_level)
            clicks.append(iter_click_list)
            observes.append(exam_p_list)
        observes, clicks = [np.array(_) for _ in (observes, clicks)]
        self.save_matrix(clicks, 'UBM')
        print(observes)
        print(clicks)

    def save_matrix(self, matrix, model='PBM'):
        np.save(os.path.join(self.data_dir, '{}.npy'.format(model)), matrix)


if __name__ == '__main__':
    program = GetClickResult(5)
    # program.generate_PBM()
    # program.generate_UBM()
    program.generate_CCM()