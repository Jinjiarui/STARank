import os
import time
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch
from torch.nn import BCELoss
from torch.nn.functional import nll_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from tqdm import tqdm

from Script.SimpleDataLoader import SimpleDataset, collate_fn_point, collate_fn_seq
from Script.options import get_options
from Script.utils import load_model, Evaluate, set_random_seed


def main(args: dict):
    print(args)
    set_random_seed(args['rand_seed'])

    # Determine whether the model output is based on points or sequences
    seq = True if args['model'] in ['Pointer'] else False

    # Create Dataset
    user_item = np.load(os.path.join(args['data_dir'], args['dataset'], 'user_item.npz'))
    click_model = None
    if args['click_model'] != '':
        click_model = np.load(os.path.join(args['data_dir'], 'ClickModel', '{}.npy'.format(args['click_model'])))
    dataset = SimpleDataset(user_item, mode='train', click_model=click_model)
    collate_fn = collate_fn_seq if seq else collate_fn_point
    dataloader = data.DataLoader(dataset, batch_size=args['batch_size'],
                                 shuffle=True, num_workers=4, collate_fn=collate_fn)
    args.update({'input_size': dataset.fields_num_sum})

    # Create Model
    model = load_model(args).to(args['device'])
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
    model_path = os.path.join(args['save_dir'], args['exp_name'])
    if args['load_model']:
        model.load_state_dict(torch.load(model_path, map_location=args['device']))
        print("Load Model from {}".format(model_path))
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['l2_reg'])
    criterion = (lambda y_, y: nll_loss(y_.log(), y)) if seq else BCELoss()
    best_loss = 1e9
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.9, patience=1000, verbose=True, min_lr=args['min_lr'])
    eva = Evaluate(criterion=criterion, mode=1 if seq else 0)
    print("-----------------Training Start-----------------\n")
    for epoch in range(args['num_epochs']):
        avg_time = 0
        eval_list = [torch.tensor([]), torch.tensor([])]
        dataset.change_mode('train')
        model.train()
        for i, (x, y) in enumerate(tqdm(dataloader)):
            t0 = time.perf_counter()
            x, y = [_.to(args['device']) for _ in [x, y]]
            y_ = model(x)
            loss = criterion(y_, y)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            eval_list[0] = torch.cat([eval_list[0], y.detach().cpu()])
            eval_list[1] = torch.cat([eval_list[1], y_.detach().cpu()])
            if eval_list[0].shape[0] > 10000:
                eval_list[0] = eval_list[0][-10000:]
                eval_list[1] = eval_list[1][-10000:]
            _, ndcgs, maps = eva.evaluate(eval_list[1], eval_list[0])
            scheduler.step(loss)
            avg_time = (avg_time * i + (time.perf_counter() - t0)) / (i + 1)
            print('Epoch:{}, batch:{}, avg_time:{:.4f} loss:{:.4f} ndcg3:{:.4f} ndcg5:{:.4f} map3:{:.4f} map5:{:.4f}'
                  .format(epoch, i, avg_time, loss, ndcgs[0], ndcgs[1], maps[0], maps[1]))
        print("-----------------Validating Start-----------------\n")
        eval_list = [[], []]
        dataset.change_mode('valid')
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(dataloader)):
                if i >= args['valid_step']:
                    break
                y_ = model(x.to(args['device'])).cpu()
                eval_list[0].append(deepcopy(y))
                eval_list[1].append(deepcopy(y_))
        eval_list = [torch.cat(_) for _ in eval_list]
        print(eval_list)
        loss, ndcgs, maps = eva.evaluate(eval_list[1], eval_list[0], True)
        print("Validate loss:{:.4f} ndcg3:{:.4f} ndcg5:{:.4f} map3:{:.4f} map5:{:.4f}"
              .format(loss, ndcgs[0], ndcgs[1], maps[0], maps[1]))
        if loss <= best_loss:
            best_loss = loss
            torch.save(model.state_dict(), model_path)
            print("New best dataset Saved!")
        print("Best Loss Now:{:.4f}".format(best_loss))
    print("-----------------Testing Start-----------------\n")
    eval_list = [[], []]
    dataset.change_mode('test')
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            y_ = model(x.to(args['device'])).cpu()
            eval_list[0].append(deepcopy(y))
            eval_list[1].append(deepcopy(y_))
    eval_list = [torch.cat(_) for _ in eval_list]
    print(eval_list)
    loss, ndcgs, maps = eva.evaluate(eval_list[1], eval_list[0], True)
    print("Test loss:{:.4f} ndcg3:{:.4f} ndcg5:{:.4f} map3:{:.4f} map5:{:.4f}"
          .format(loss, ndcgs[0], ndcgs[1], maps[0], maps[1]))


if __name__ == '__main__':
    torch.set_num_threads(6)
    parser = ArgumentParser("Set-to-Sequence")
    args = get_options(parser)
    main(args)
