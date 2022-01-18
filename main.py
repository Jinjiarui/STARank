import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import BCELoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from tqdm import tqdm

from Script.SimpleDataLoader import SimpleDataset, collate_fn
from Script.options import get_options
from Script.utils import load_model, evaluate_utils, set_random_seed


def main(args: dict):
    print(args)

    set_random_seed(args['rand_seed'])
    # Create Dataset
    user_item = np.load(os.path.join(args['data_dir'], args['dataset'], 'user_item.npz'))
    dataset = SimpleDataset(user_item, mode='train')
    dataloader = data.DataLoader(dataset, batch_size=args['batch_size'], collate_fn=collate_fn,
                                 shuffle=True, num_workers=4)
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
    criterion = BCELoss()
    best_auc = 0
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.9, patience=1000, verbose=True, min_lr=args['min_lr'])

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
            if eval_list[0].shape[0] > 100000:
                eval_list[0] = eval_list[0][-100000:]
                eval_list[1] = eval_list[1][-100000:]
            _, acc, auc = evaluate_utils(eval_list[1], eval_list[0])
            scheduler.step(auc)
            avg_time = (avg_time * i + (time.perf_counter() - t0)) / (i + 1)
            print('Epoch:{}, batch:{}, avg_time:{:.4f} loss:{:.4f} acc:{:.4f} auc:{:.4f}'
                  .format(epoch, i, avg_time, loss, acc, auc))
        print("-----------------Validating Start-----------------\n")
        eval_list = [[], []]
        dataset.change_mode('valid')
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(dataloader)):
                if i >= args['valid_step']:
                    break
                y_ = model(x.to(args['device'])).cpu()
                eval_list[0].append(y)
                eval_list[1].append(y_)
        eval_list = [torch.cat(_) for _ in eval_list]
        loss, acc, auc = evaluate_utils(eval_list[1], eval_list[1], criterion)
        print("Validate loss:{:.4f} acc:{:.4f} auc:{:.4f}"
              .format(loss, acc, auc))
        if auc >= best_auc:
            best_auc = auc
            torch.save(model.state_dict(), model_path)
            print("New best dataset Saved!")
        print("Best Auc Now:{}".format(best_auc))
    print("-----------------Testing Start-----------------\n")
    eval_list = [[], []]
    dataset.change_mode('test')
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            y_ = model(x.to(args['device'])).cpu()
            eval_list[0].append(y)
            eval_list[1].append(y_)
    eval_list = [torch.cat(_) for _ in eval_list]
    loss, acc, auc = evaluate_utils(eval_list[1], eval_list[1], criterion)
    print("Test loss:{:.4f} acc:{:.4f} auc:{:.4f}"
          .format(loss, acc, auc))


if __name__ == '__main__':
    parser = ArgumentParser("Set-to-Sequence")
    args = get_options(parser)
    main(args)
