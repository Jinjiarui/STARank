import random


def cal_binary_relevance(relevance, relevance_level):
    """transform form 5 relevance level into binary relevance"""
    binary_relevance = 0.1 + 0.9 * (pow(2, relevance) - 1) / (pow(2, relevance_level - 1) - 1)
    return binary_relevance


def cal_discrete_relevance(relevance, relevance_level):
    discrete_relevance = 1 if relevance >= relevance_level / 2 else 0
    return discrete_relevance


class ClickModel:
    def __init__(self):
        self.click_pro = []
        self.observe_pro = []
        self.observe_pro_list = []

    def set_click_probability(self, relevance, relevance_level, position):
        raise NotImplementedError

    def set_observe_probability(self):
        raise NotImplementedError

    def get_click_probability(self, relevance, relevance_level, position):
        # self.set_observe_probability()
        self.set_click_probability(relevance, relevance_level, position)
        return self.observe_pro, self.click_pro


class PositionBiasedModel(ClickModel):
    def set_observe_probability(self, i=None):
        observe_pro_mean = [1, 1, 0.95, 0.82, 0.69, 0.54, 0.47, 0.43, 0.41, 0.43]
        # observe_pro_var = [0, 0, 0.01, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03, 0.04]
        if i is not None:
            observe_pro_mean = [i, 1, 0.95, 0.82, 0.69, 0.54, 0.47, 0.43, 0.41, 0.43]
        self.observe_pro_list = observe_pro_mean

    def set_click_probability(self, relevance, relevance_level, position):

        binary_relevance = cal_binary_relevance(relevance, relevance_level)
        # return click value
        if position < 10:
            self.observe_pro = self.observe_pro_list[position]
            self.click_pro = binary_relevance * self.observe_pro
        else:
            self.observe_pro = 0.43
            self.click_pro = binary_relevance * 0.43


class ClickClainModel(ClickModel):
    def __init__(self, non_observe, click_observe):
        self.gama_2 = non_observe  # gama 3
        self.gama_3 = click_observe
        self.gama_1 = 0.5

        # self.non_observe_alpha = non_observe
        # self.click_observe_alpha = click_observe
        super(ClickClainModel, self).__init__()

    def set_browse_probability(self, last_observe, last_click, relevance, relevance_level):
        # last_observe and last_click are binary number
        binary_relevance = cal_binary_relevance(relevance, relevance_level)
        self.observe_pro = last_click * last_observe * (
                binary_relevance * self.gama_3 + (1 - binary_relevance) * self.gama_2) + last_observe * (
                                   1 - last_click) * self.gama_1
        # self.observe_pro = last_click * last_observe * (binary_relevance * self.click_observe_alpha) + last_observe * (1 - last_click) * self.non_observe_alpha

        self.click_pro = self.observe_pro * binary_relevance
        return self.observe_pro, self.click_pro


class UserBrowsingModel(ClickModel):

    def set_observe_probability(self):
        observe_pro_mean = [
            [1.0],
            [0.98, 1.0],
            [1.0, 0.62, 0.95],
            [1.0, 0.77, 0.42, 0.82],
            [1.0, 0.92, 0.55, 0.31, 0.69],
            [1.0, 0.96, 0.63, 0.4, 0.22, 0.54],
            [1.0, 0.99, 0.73, 0.46, 0.29, 0.17, 0.47],
            [1.0, 1.0, 0.89, 0.52, 0.35, 0.24, 0.14, 0.43],
            [1.0, 1.0, 0.95, 0.68, 0.4, 0.29, 0.19, 0.12, 0.41],
            [1.0, 1.0, 1.0, 0.96, 0.52, 0.36, 0.27, 0.18, 0.12, 0.43]
        ]
        observe_pro_var = [
            [0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.03, 0.03, 0.04, 0.03, 0.01, 0.01, 0, 0],
            [0.01, 0.02, 0.02, 0.05, 0.09, 0.07, 0.04, 0.01],
            [0.03, 0.02, 0.03, 0.03, 0.05, 0.1, 0.08],
            [0.03, 0.02, 0.02, 0.03, 0.03, 0.06],
            [0.03, 0.01, 0.02, 0.02, 0.04],
            [0.02, 0.01, 0.02, 0.03],
            [0.03, 0.01, 0.02],
            [0.03, 0.02],
            [0.04]
        ]

        # use mean for observe_pro
        self.observe_pro_list = observe_pro_mean

    def sampleClicksForOneList(self, label_list, relevance_level=5):
        click_list, exam_p_list = [], []
        last_click_rank = -1
        for rank in range(len(label_list)):
            click, exam_p = self.sampleClick(rank, last_click_rank, label_list[rank], relevance_level)
            if click > 0:
                last_click_rank = rank
            click_list.append(click)
            exam_p_list.append(exam_p)
        return click_list, exam_p_list

    def sampleClick(self, rank, last_click_rank, relevance_label, relevance_level=5):
        if not relevance_label == int(relevance_label):
            print('RELEVANCE LABEL MUST BE INTEGER!')
        binary_relevance = cal_binary_relevance(relevance_label, relevance_level)
        exam_p = self.getExamProb(rank, last_click_rank)
        click = 1 if random.random() < exam_p * binary_relevance else 0
        return click, exam_p

    def sample_Clicks_and_Observes_ForOneList(self, label_list, relevance_level=5):
        click_list, exam_p_list = [], []
        last_click_rank = -1
        for rank in range(len(label_list)):
            click, exam_p = self.sample_click_and_observe(rank, last_click_rank, label_list[rank], relevance_level)
            if click > 0:
                last_click_rank = rank
            click_list.append(click)
            exam_p_list.append(exam_p)
        return click_list, exam_p_list

    def sample_click_and_observe(self, rank, last_click_rank, relevance_label, relevance_level=5):
        if not relevance_label == int(relevance_label):
            print('RELEVANCE LABEL MUST BE INTEGER!')
        binary_relevance = cal_binary_relevance(relevance_label, relevance_level)
        exam_p = self.getExamProb(rank, last_click_rank)
        random_var = random.random()
        click = 1 if random_var < exam_p * binary_relevance else 0
        observe = 1 if random_var < exam_p else 0
        return click, observe

    def getExamProb(self, rank, last_click_rank):
        distance = rank - last_click_rank
        if rank < len(self.observe_pro_list):
            exam_p = self.observe_pro_list[rank][distance - 1]
        else:
            if distance > rank:
                exam_p = self.observe_pro_list[-1][-1]
            else:
                idx = distance - 1 if distance < len(self.observe_pro_list[-1]) - 1 else -2
                exam_p = self.observe_pro_list[-1][idx]
        return exam_p


class RelevanceModel:
    def __init__(self):
        self.observe_pro = []
        self.relevance_pro = []
