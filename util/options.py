import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_exp', type=int, default=1, help='number of experiments')
    parser.add_argument('--print_txt', type=bool, default=True, help='print txt')
    parser.add_argument('--num_classes', type=int, default=7, help='number of classes')
    parser.add_argument('--lr_w', type=float, default=0.01, help='learning rate for warming up')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for training')
    parser.add_argument('--lr_f', type=float, default=1e-3, help='learning rate for fine-tuning')
    parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
    parser.add_argument('--noise_rate', type=float, help='overall corruption rate', default=0.4)
    parser.add_argument('--dataset', type=str, help='ham10000, aptos', default='ham10000')
    parser.add_argument('--round1', type=int, help='number of rounds for warming up', default=1)
    parser.add_argument('--round2', type=int, help='number of rounds for training transition matrix estimation', default=50)
    parser.add_argument('--round3', type=int, help='number of rounds for fine-tuning', default=50)
    parser.add_argument('--local_ep', type=int, help='number of local epochs', default=1)
    parser.add_argument('--local_ep2', type=int, help='number of local epochs', default=1)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_workers', type=int, help='how many subprocesses to use for data loading', default=16)
    parser.add_argument('--gpu', type=int, help='ind of gpu', default=1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=int, help='momentum', default=0.9)
    parser.add_argument('--batch_size', type=int, help='batch_size', default=16)
    parser.add_argument('--split_percentage', type=float, help = 'train and validation', default=0.90)
    parser.add_argument('--tau', type=float, help='threshold', default=0.5)
    parser.add_argument('--mixhigh', type=float, help='threshold', default=0.5)
    parser.add_argument('--mixlow', type=float, help='threshold', default=0.5)
    parser.add_argument('--mu', type=float, help='threshold', default=0.01)
    parser.add_argument('--num_clients', type=int, help = 'number of total clients', default=10)
    parser.add_argument('--frac', type=float, default=1.0, help="fraction of selected clients")
    parser.add_argument('--iid', type=bool, help='i.i.d. or non-i.i.d.', default=True)
    
    # ROFL
    parser.add_argument('--forget_rate', type=float, default=0.2, help="forget rate")
    parser.add_argument('--num_gradual', type=int, default=10, help='T_k')
    parser.add_argument('--T_pl', type=int, help = 'T_pl: When to start using global guided pseudo labeling', default=100)
    parser.add_argument('--lambda_cen', type=float, help = 'lambda_cen', default=1.0)
    parser.add_argument('--lambda_e', type=float, help = 'lambda_e', default=0.8)
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--num_shards', type=int, default=200, help="number of shards")
    parser.add_argument('--schedule', nargs='+', default=[], help='decrease learning rate at these epochs.')
    parser.add_argument('--feature_dim', type=int, help = 'feature dimension', default=384)

    #ARFL
    parser.add_argument('--local_iter', type=int, default=30, help="local iteration(number of batch)")
    parser.add_argument('--use_memory', type=str2bool, default=True, help="use FoolsGold memory option")
    parser.add_argument('--verbose', type=int, default=0, help='verbose print, 1 for True, 0 for False')
    parser.add_argument('--Lambda', type=float, default=2.0, help='set lambda of irls (default: 2.0)')
    parser.add_argument('--thresh', type=float, default=0.1, help='set thresh of irls restriction (default: 0.1)')
    parser.add_argument('--alpha', type=float, default=0.2, help='set thresh of trimmed mean (default: 0.2)')
    parser.add_argument('--agg', type=str, default='average', choices=['average', 'median', 'trimmed_mean',
                                                                           'repeated', 'irls', 'simple_irls',
                                                                           'irls_median', 'irls_theilsen',
                                                                           'irls_gaussian', 'fg'])
    
                        
    return parser.parse_args()