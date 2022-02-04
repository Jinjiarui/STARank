import os

import torch
from torch.nn.functional import binary_cross_entropy
from torch import optim
from tqdm import tqdm

import generate_data
from pointer_network import PointerNetwork
from utils import to_var

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
total_size = 10000
weight_size = 256
emb_size = 32
batch_size = 250
n_epochs = 200

input_seq_len = 20
inp_size = input_seq_len * 6

inputs, answers, target_diff, targets = generate_data.make_seq_reward_data(total_size, input_seq_len, inp_size)

# Convert to torch tensors
inputs = to_var(torch.LongTensor(inputs))  # (N, L)
answers = to_var(torch.LongTensor(answers))  # (N, 2)
targets = to_var(torch.tensor(targets).view(-1, 1))  # (N,1)
rewards = to_var(torch.tensor(target_diff, dtype=torch.float32))  # (N)
print(inputs)
print(answers)
print(targets)
print(rewards)

data_split = int(total_size * 0.5)
train_X = inputs[:data_split]
train_Y = answers[:data_split]
train_T = targets[:data_split]
train_R = rewards[:data_split]
test_X = inputs[data_split:]
test_Y = answers[data_split:]
test_T = targets[data_split:]
test_R = rewards[data_split:]
# cri = lambda y_, y, r: -torch.gather(torch.log(y_), -1, y).squeeze() * r.view((-1, 1))
cri = lambda y_, y, r: binary_cross_entropy(torch.gather(y_, -1, y).squeeze(),
                                            r.view((-1, 1)).expand((-1, y.size(1))))


# from pointer_network import PointerNetwork
def train(model, X, Y, T, R, batch_size, n_epochs):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    N = X.size(0)
    L = X.size(1)
    for epoch in tqdm(range(n_epochs + 1)):
        losses = []
        for i in range(0, N - batch_size, batch_size):
            x = X[i:i + batch_size]  # (bs, L)
            t = T[i:i + batch_size]  # (bs, 1)
            # r = R[i:i + batch_size]
            # y = Y[i:i + batch_size]

            probs = model(torch.stack([x, t.expand(-1, L)], dim=-1))  # (bs, M, L)
            # y_ = torch.argmax(probs, -1)  # (bs, M)
            y_ = torch.multinomial(probs.view(-1, L), 1).view(batch_size, -1)
            # y_ = torch.randint_like(probs[:, :, 0], 0, L, dtype=torch.int64)
            r = -torch.mean(torch.abs(torch.gather(x, -1, y_).float() - t), dim=-1)  # （bs,）
            r = torch.where(r > -35, torch.ones_like(r), torch.zeros_like(r))
            # r = (r - r.min()) / (torch.std(r))
            loss = torch.mean(cri(probs, y_.unsqueeze(-1), r))
            losses.append(loss.item())
            # y = torch.argsort(torch.abs(x.float() - t))[:, :probs.size(1)]
            # loss = f.nll_loss(probs.view(-1, L), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses = torch.mean(torch.tensor(losses))
        if epoch % 2 == 0:
            print('epoch: {}, Loss: {:.5f}'.format(epoch, losses.item()))
            test(model, X, R, T)


def test(model, X, R, T):
    with torch.no_grad():
        probs = model(torch.stack([X, T.expand(-1, X.size(1))], dim=-1))  # (bs, M, L)
    indices = torch.argmax(probs, -1)  # (bs, M)
    rewards_test = -torch.mean(torch.abs(torch.gather(X, -1, indices).float() - T), dim=-1)

    rewards_best = -torch.abs(X.float() - T)
    rewards_best = torch.mean(torch.sort(rewards_best, dim=-1)[0][:, -10:], dim=-1)

    print(rewards_test.min().item(), rewards_test.max().item(), torch.std(rewards_test).item())
    print(rewards_best.min().item(), rewards_best.max().item(), torch.std(rewards_best).item())
    print(
        'Old Reward:{:.4f} New Reward:{:.4f} Best Reward:{:.4f}'
            .format(torch.mean(R), torch.mean(rewards_test), torch.mean(rewards_best)))


model = PointerNetwork(inp_size, emb_size, weight_size, 10, multi_field=True, is_GRU=True)
if torch.cuda.is_available():
    model.cuda()
train(model, train_X, train_Y, train_T, train_R, batch_size, n_epochs)
print('----Test result---')
test(model, test_X, test_R, test_T)
