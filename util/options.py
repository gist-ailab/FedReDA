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
    parser = argparse.ArgumentParser(description="Unified Runner for FedLNL and other FL models")

    # ========================= General Arguments =========================
    group_general = parser.add_argument_group('General')
    group_general.add_argument('--num_exp', type=int, default=1, help='number of experiments')
    group_general.add_argument('--print_txt', type=bool, default=True, help='print txt')
    group_general.add_argument('--num_classes', type=int, default=5, help='number of classes')
    group_general.add_argument('--lr', type=float, default=3e-4, help='learning rate for training')
    group_general.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
    group_general.add_argument('--dataset', type=str, help='ich, ham10000, aptos, cifar10, cifar100', default='ich')
    group_general.add_argument('--seed', type=int, default=0)
    group_general.add_argument('--print_freq', type=int, default=100)
    group_general.add_argument('--num_workers', type=int, help='how many subprocesses to use for data loading', default=16)
    group_general.add_argument('--gpu', type=int, help='ind of gpu', default=0)
    group_general.add_argument('--weight_decay', type=float, default=5e-4)
    group_general.add_argument('--momentum', type=int, help='momentum', default=0.9)
    group_general.add_argument('--batch_size', type=int, help='batch_size', default=16)
    group_general.add_argument('--num_clients', type=int, help = 'number of total clients', default=20)
    group_general.add_argument('--iid', type=bool, help='i.i.d. or non-i.i.d.', default=0)
    group_general.add_argument('--frac', type=float, default=1.0, help="fraction of selected clients")
    parser.add_argument('--warmup_keep_ratio', type=float, default=0.8,
                    help='Ratio of small-loss samples to keep during warm-up')

    # ========================= FedLNL Arguments =========================
    group_fedlnl = parser.add_argument_group('FedLNL')
    group_fedlnl.add_argument('--lr_f', type=float, default=1e-3, help='[FedLNL] learning rate for fine-tuning')
    group_fedlnl.add_argument('--round1', type=int, help='[FedLNL] number of rounds for warming up', default=1)
    group_fedlnl.add_argument('--round2', type=int, help='[FedLNL] number of rounds for training transition matrix estimation', default=50)
    group_fedlnl.add_argument('--round3', type=int, help='[FedLNL] number of rounds for fine-tuning', default=50)
    group_fedlnl.add_argument('--local_ep', type=int, help='number of local epochs', default=5)
    group_fedlnl.add_argument('--tau', type=float, help='[FedLNL] threshold', default=0.5)
    group_fedlnl.add_argument('--mixhigh', type=float, help='[FedLNL] mixup alpha high', default=0.5)
    group_fedlnl.add_argument('--mixlow', type=float, help='[FedLNL] mixup alpha low', default=0.1)
    group_fedlnl.add_argument('--mu', type=float, help='[FedLNL] PAPA mu', default=0.01)
    group_fedlnl.add_argument('--feature_dim', type=int, help = 'feature dimension', default=384)

    # ========================= FedNoRo Arguments =========================
    group_fednoro = parser.add_argument_group('FedNoRo')
    group_fednoro.add_argument('--non_iid_prob_class', type=float, default=0.9, help='[FedNoRo] parameter for non-iid class distribution')
    group_fednoro.add_argument('--alpha_dirichlet', type=float, default=2.0, help='[FedNoRo] parameter for non-iid dirichlet distribution')
    group_fednoro.add_argument('--level_n_system', type=float, default=1.0, help='[FedNoRo] system-level noise rate')
    group_fednoro.add_argument('--level_n_lowerb', type=float, default=0.3, help='[FedNoRo] lower bound of client-level noise')
    group_fednoro.add_argument('--level_n_upperb', type=float, default=0.5, help='[FedNoRo] upper bound of client-level noise')
    group_fednoro.add_argument('--n_parties', type=int, default=10, help='[FedNoRo] number of participating parties')
    group_fednoro.add_argument('--n_type', type=str, default='instance', help='[FedNoRo] type of noise, instance or symmetric')
    group_fednoro.add_argument('--rounds', type=int, default=100, help='[FedNoRo] total communication rounds')
    group_fednoro.add_argument('--s1', type=int, default=10, help='[FedNoRo] rounds for stage 1 (warm-up)')
    group_fednoro.add_argument('--warm', type=str2bool, default=True, help='[FedNoRo] whether to use warm-up stage')
    group_fednoro.add_argument('--a', type=float, default=0.8, help='[FedNoRo] weight of consistency regularization')
    group_fednoro.add_argument('--begin', type=int, default=10, help='[FedNoRo] when to begin consistency regularization')
    group_fednoro.add_argument('--end', type=int, default=49, help='[FedNoRo] when to end consistency regularization')
    group_fednoro.add_argument('--deterministic', type=str2bool, default=True, help='[FedNoRo] whether to use deterministic training')

    # ========================= ROFL Arguments =========================
    group_rofl = parser.add_argument_group('ROFL')
    group_rofl.add_argument('--forget_rate', type=float, default=0.2, help="[ROFL] forget rate")
    group_rofl.add_argument('--num_gradual', type=int, default=10, help='[ROFL] T_k')
    group_rofl.add_argument('--T_pl', type=int, help = '[ROFL] When to start using global guided pseudo labeling', default=100)
    group_rofl.add_argument('--lambda_cen', type=float, help = '[ROFL] lambda_cen', default=1.0)
    group_rofl.add_argument('--lambda_e', type=float, help = '[ROFL] lambda_e', default=0.8)

    # ========================= ARFL Arguments =========================
    group_arfl = parser.add_argument_group('ARFL')
    group_arfl.add_argument('--local_iter', type=int, default=30, help="[ARFL] local iteration(number of batch)")
    group_arfl.add_argument('--use_memory', type=str2bool, default=True, help="[ARFL] use FoolsGold memory option")
    group_arfl.add_argument('--verbose', type=int, default=0, help='[ARFL] verbose print, 1 for True, 0 for False')
    group_arfl.add_argument('--Lambda', type=float, default=2.0, help='[ARFL] set lambda of irls (default: 2.0)')
    group_arfl.add_argument('--thresh', type=float, default=0.1, help='[ARFL] set thresh of irls restriction (default: 0.1)')
    group_arfl.add_argument('--alpha', type=float, default=0.2, help='[ARFL] set thresh of trimmed mean (default: 0.2)')
    group_arfl.add_argument('--agg', type=str, default='average', choices=['average', 'median', 'trimmed_mean',
                                                                           'repeated', 'irls', 'simple_irls',
                                                                           'irls_median', 'irls_theilsen',
                                                                           'irls_gaussian', 'fg'])
    
    return parser.parse_args()