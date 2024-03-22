import argparse
import configparser

# *****************************************  参数初始化配置 ****************************************** #
Mode = 'train'
DEBUG = 'False'
DATASET = 'PEMS04'
DEVICE = 'cuda:1'
MODEL = 'AFDGCN'
GRAPH = "./data/PEMS04/PEMS04.csv"
K = 0.
# FILENAME_ID = "./data/PEMS03/PEMS03.txt"
FILENAME_ID = None

# 1. get configuration
config_file = './conf/{}_{}.conf'.format(DATASET, MODEL)
config = configparser.ConfigParser()
config.read(config_file)

# 2. parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)
args.add_argument('--graph_path', default=GRAPH, type=str)
args.add_argument('--normalized_k', default=K, type=float)
args.add_argument('--filename_id', default=FILENAME_ID, type=str)
# 3. data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
# 4. train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
args.add_argument('--real_value', default=config['train']['real_value'], type=eval)
# 6. test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--rmse_thresh', default=config['test']['rmse_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
# 7. log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument("--checkpoint", type=str, default="./save_model/", help="pre-trained model file")
args = args.parse_args()
# *****************************************  参数初始化配置 ****************************************** #
