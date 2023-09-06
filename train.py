import os
import torch
import numpy as np
import torch.nn as nn
from config import args
from datetime import datetime
import torch.nn.functional as F
from model.AFDGCN import Model as Network
from engine import Engine
from lib.metrics import MAE_torch
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.load_graph import get_Gaussian_matrix, get_adjacency_matrix

# *****************************************  参数初始化配置 ****************************************** #
init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'
# load dataset
# A = get_adjacency_matrix(args.graph_path, args.num_nodes, type='connectivity', id_filename=args.filename_id)
A = get_Gaussian_matrix(args.graph_path, args.num_nodes, args.normalized_k, id_filename=args.filename_id)
A = torch.FloatTensor(A).to(args.device)
train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, 
                                                               dow=False,
                                                               weather=False, 
                                                               single=False)
# *****************************************  初始化模型参数 ****************************************** #
input_dim = 1
hidden_dim = 64 
output_dim = 1
embed_dim = 8   
cheb_k = 2
horizon = 12 
num_layers = 1 
heads = 4
timesteps = 12
kernel_size = 5
model = Network(num_node = args.num_nodes, 
                input_dim = input_dim, 
                hidden_dim = hidden_dim, 
                output_dim = output_dim, 
                embed_dim = embed_dim, 
                cheb_k = cheb_k, 
                horizon = horizon, 
                num_layers = num_layers, 
                heads = heads, 
                timesteps = timesteps, 
                A = A,
                kernel_size=kernel_size)
model = model.to(args.device)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model.cuda(), device_ids=[3, 4])
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
# print the number of model parameters
print_model_parameters(model, only_num=False)
# *****************************************  定义损失函数、优化器 ****************************************** #
# 1. init loss function, optimizer
def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss
if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
elif args.loss_func == 'smoothloss':
    loss = torch.nn.SmoothL1Loss().to(args.device)
else:
    raise ValueError
# 2. init optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
# 3. learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
# *****************************************  模型训练与测试 ****************************************** #
# 1.config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'experiments', args.dataset, current_time)
print(log_dir)
args.log_dir = log_dir

# 2.start training
trainer = Engine(model, 
                  loss, 
                  optimizer, 
                  train_loader, 
                  val_loader, 
                  test_loader, 
                  scaler,
                  args, 
                  lr_scheduler)
if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
    checkpoint = "./experiments/PEMS04/20220811215513/PEMS04_AFDGCN_best_model.pth"
    model.load_state_dict(torch.load(checkpoint))  # map_location='cuda:5'
    # node_embedding = model.node_embedding
    # adj = F.softmax(F.relu(torch.mm(node_embedding, node_embedding.transpose(0, 1))), dim=1)
    # adj = torch.mm(node_embedding, node_embedding.transpose(0, 1))
    # print(adj.shape)
    # np.save('adaptive_matrix.npy', adj.detach().cpu().numpy())
    print("load saved model...")
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
else:
    raise ValueError