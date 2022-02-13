import os

import numpy as np

from Script.Click import PositionBiasedModel


class GetClickResult:
    def __init__(self, data_dir='./Data', name='tmall'):
        self.dataset_name = name
        self.data_dir = os.path.join(data_dir, name)
        self.init_relevance = np.load(os.path.join(self.data_dir, 'user_item.npz'))['log'][:, :, -1]
        self.relevance_level = np.unique(self.init_relevance).size

    def generate_PBM(self):
        self.dele = []
        click_model = PositionBiasedModel()
        click_model.set_observe_probability()
        clicks, observes = [], []
        for relevance in self.init_relevance:
            iter_observe_list = []
            iter_click_list = []
            for _p, _r in enumerate(relevance):
                iter_observe, iter_click = click_model.get_click_probability(relevance=_r,
                                                                             relevance_level=self.relevance_level,
                                                                             position=_p)
                iter_observe_list.append(iter_observe)
                iter_click_list.append(iter_click)
            observes.append(iter_observe_list)
            clicks.append(iter_click_list)
        observes, clicks = [np.array(_) for _ in (observes, clicks)]
        print(observes, observes.mean(), observes.min(), observes.max())
        print(clicks, clicks.mean(), clicks.min(), clicks.max())


if __name__ == '__main__':
    program = GetClickResult(name='alipay')
    program.generate_PBM()
