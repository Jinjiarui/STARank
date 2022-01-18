from argparse import ArgumentParser


def get_exp_configure(args):
    dataset_list = {'tmall': {}, 'alipay': {}, 'taobao': {}}
    model_list = ['LSTM', 'GRU']
    for dataset in dataset_list.keys():
        for model in model_list:
            dataset_list[dataset][model] = {
                'embed_size': 48,
                'hidden_size': 64,
                'output_size': 1,
                'batch_size': 250,
                'dropout_rate': 0.5,
                'decay_step': 1000,
                'min_lr': 1e-5,
                'l2_reg': 4e-5,
                'predict_hidden_sizes': [256, 64, 16],
            }
    return dataset_list[args['dataset']][args['model']]


def get_options(parser: ArgumentParser, reset_args=None):
    from torch import device
    if reset_args is None:
        reset_args = {}
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['tmall', 'alipay', 'taobao'],
                        default='tmall', help='Dataset use')
    parser.add_argument('-m', '--model', type=str,
                        choices=['LSTM', 'GRU', ],
                        default='LSTM', help='Model use'
                        )
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--save_dir', type=str, default='./SavedModels')
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('-c', '--cuda', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--valid_step', type=int, default=200)
    parser.add_argument('--postfix', type=str, default='',
                        help="a string appended to the file name of the saved model")
    parser.add_argument('--rand_seed', type=int, default=-1, help="random seed for torch and numpy")
    parser.set_defaults(**reset_args)

    args = parser.parse_args().__dict__
    # Get experiment configuration
    args['exp_name'] = '_'.join([args['model'], args['dataset'], args['postfix']])
    args.update(get_exp_configure(args))

    device_name = 'cpu' if args['cuda'] < 0 else 'cuda:{}'.format(args['cuda'])
    args['device'] = device(device_name)
    return args
