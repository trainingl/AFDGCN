# AFDGCN
Code and logs of  our paper “Dynamic Graph Convolutional Network with Attention Fusion for Traffic Flow Prediction” accepted by ECAI 2023.

## 😏Requirements
Install the version of the following python module：
```bash
numpy==1.23.5
pandas==1.1.3
torch==1.8.0
tqdm==4.50.2
```

## ⚙Training
If you want to reproduce the experimental results of the model, read the parameter item settings in the experiment and run `python train.py`.
```python
# *****************************************  初始化模型参数 ****************************************** #
input_dim = 1
hidden_dim = 64 
output_dim = 1
embed_dim = 8    # PEMSD4: 4
cheb_k = 2
horizon = 12 
num_layers = 1 
heads = 4   
timesteps = 12
kernel_size = 5  # PEMSD8: 9
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
# *****************************************  初始化模型参数 ****************************************** #
```
## ✨Acknowledgments
Thanks to the authors of [STG-NCDE](https://github.com/jeongwhanchoi/STG-NCDE) for providing us with their source code and experimental results (retained models and logs) for visual analysis in our results.


