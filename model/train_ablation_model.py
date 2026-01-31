"""
扩展的训练脚本，支持消融实验参数
基于 train_best_model.py 修改，添加了更多控制参数用于消融实验
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from all_models import *
from data_processing import *
from torch.utils.data import DataLoader
from defined_functions import *
import math
from datetime import datetime
from torch.nn import functional as F
import argparse
import json
from evaluation import *
from tqdm import tqdm
from comparing_models import *
from torch.optim.lr_scheduler import MultiStepLR
import click as ck
from multiprocessing import Pool
from functools import partial
from dataset_generating.basics import propagate_annots
import random
import os


def set_seed(seed=42):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


@ck.command()
# 原始参数
@ck.option('--aspect', '-asp', default='mf', type=ck.Choice(['mf', 'cc', 'bp']))
@ck.option('--deepgo2', '-dp', is_flag=True)
@ck.option('--fastdataloader', '-fdl', is_flag=True)
@ck.option('--protein_embedding_path', '-ep', default='../data/esm_embeddings_3B_complete_2021&2025.pt')
@ck.option('--go_embedding_path', '-ge', default='../data/go_all_embeddings.pt')
@ck.option('--seed', '-s', default=42, type=int)
@ck.option('--loss', '-l', default='focal', type=ck.Choice(['bce', 'focal', 'asymmetric', 'combined']))
@ck.option('--temperature', '-t', is_flag=True)

# 新增: 消融实验参数
@ck.option('--num_heads', '-nh', default=None, type=int, help='Number of attention heads (override config)')
@ck.option('--num_layers', '-nl', default=None, type=int, help='Number of attention layers (override config)')
@ck.option('--alpha_value', '-alpha', default=None, type=float, help='Fixed alpha value (if set, alpha becomes non-learnable)')
@ck.option('--learnable_alpha/--fixed_alpha', default=True, help='Whether alpha is learnable')
@ck.option('--use_residual', default=True, type=bool, help='Use residual connections')
@ck.option('--use_go_bias', default=True, type=bool, help='Use GO bias term')
@ck.option('--freeze_go_embedding', is_flag=True, help='Freeze GO embedding (not learnable)')
@ck.option('--random_go_embedding', is_flag=True, help='Randomly initialize GO embedding')
@ck.option('--dropout', '-drop', default=0.1, type=float, help='Dropout rate')
@ck.option('--lr', default=5e-4, type=float, help='Learning rate')
@ck.option('--scheduler', default='step', type=ck.Choice(['step', 'cosine', 'plateau', 'none']), help='LR scheduler')
@ck.option('--exp_id', default=None, type=str, help='Experiment ID for logging')

def main(aspect, deepgo2, fastdataloader, protein_embedding_path, go_embedding_path,
         seed, loss, temperature, num_heads, num_layers, alpha_value, learnable_alpha,
         use_residual, use_go_bias, freeze_go_embedding, random_go_embedding,
         dropout, lr, scheduler, exp_id):

    # 设置随机种子
    set_seed(seed)
    print(f'Random seed set to: {seed}')

    # 实验ID用于记录
    if exp_id is None:
        exp_id = f"{aspect}_{loss}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f'Experiment ID: {exp_id}')

    # 路径设置
    if deepgo2:
        onto_path = '../../deepgo2/data/go.obo'
        data_root = '../../deepgo2/data'
    else:
        onto_path = '../data/go.obo'
        data_root = '../data/dataset/'

    # 加载配置
    with open("./config/model_config.json", "r") as f:
        _cfg = json.load(f)
        config = _cfg['profiles']
    profile = config['default']

    # 从配置或命令行获取参数
    epoch_num = profile['epoch_num']
    TRAIN_BS = profile['train_bs']
    EVAL_BS = profile['eval_bs']
    num_heads = num_heads if num_heads is not None else profile['num_heads']
    num_layers = num_layers if num_layers is not None else profile['num_layers']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*80}")
    print("实验配置:")
    print(f"{'='*80}")
    print(f"Aspect: {aspect}")
    print(f"Loss: {loss}")
    print(f"Num heads: {num_heads}")
    print(f"Num layers: {num_layers}")
    print(f"Alpha: {'learnable' if learnable_alpha else f'fixed={alpha_value}'}")
    print(f"Temperature: {temperature}")
    print(f"Use residual: {use_residual}")
    print(f"Use GO bias: {use_go_bias}")
    print(f"Freeze GO embedding: {freeze_go_embedding}")
    print(f"Random GO embedding: {random_go_embedding}")
    print(f"Dropout: {dropout}")
    print(f"Learning rate: {lr}")
    print(f"Scheduler: {scheduler}")
    print(f"{'='*80}\n")

    # 加载数据
    if deepgo2:
        protein_labels, protein_embeddings_all = load_deepgo2_data(data_root, 'mf')
    else:
        embedding_data = torch.load(protein_embedding_path)
        protein_labels, _, protein_embeddings_all = embedding_data.values()
        protein_labels = protein_labels.tolist()
    protein_embeddings_all = protein_embeddings_all.to(device)

    # 加载GO嵌入
    go_data = torch.load(go_embedding_path, weights_only=False)
    go_list = go_data["terms"]
    go_context_embeddings = go_data["context_embeddings"]
    go_embeddings = go_data["go_embeddings"]

    # 加载本体
    go = Ontology(onto_path, with_rels=True)
    model_save_file_path = f'../data/model_checkpoint/ablation_{exp_id}.pt'
    train_df, valid_df, test_df, terms, terms_dict, termidx = load_data(aspect, data_root)
    term_idxes = [idx for term, idx in terms_dict.items() if term in go_list]

    go_context_embeddings = torch.from_numpy(go_context_embeddings[term_idxes]).float().to(device)
    go_embeddings = torch.from_numpy(go_embeddings[term_idxes]).float().to(device)

    # 随机初始化GO嵌入 (消融实验)
    if random_go_embedding:
        print("⚠️  Using randomly initialized GO embeddings")
        go_embeddings = torch.randn_like(go_embeddings)

    # 构建数据集
    train_dataset = build_dataset(train_df, terms_dict, protein_embeddings_all, protein_labels, fdl=fastdataloader)
    val_dataset = build_dataset(valid_df, terms_dict, protein_embeddings_all, protein_labels, fdl=fastdataloader)
    test_dataset = build_dataset(test_df, terms_dict, protein_embeddings_all, protein_labels, fdl=fastdataloader)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        _ = worker_id

    g = torch.Generator()
    g.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BS, shuffle=True,
                                  worker_init_fn=seed_worker, generator=g)
    test_dataloader = DataLoader(test_dataset, batch_size=EVAL_BS, shuffle=False,
                                 worker_init_fn=seed_worker, generator=g)
    valid_dataloader = DataLoader(val_dataset, batch_size=EVAL_BS, shuffle=False,
                                  worker_init_fn=seed_worker, generator=g)

    train_labels = train_dataset.annotations
    valid_labels = val_dataset.annotations
    test_labels = test_dataset.annotations

    # 初始化模型 (带消融实验参数)
    combine_model = Combine_Transformer_Ablation(
        num_heads=num_heads,
        go_context=go_context_embeddings,
        embedding_vector=go_embeddings,
        num_layers=num_layers,
        device=device,
        dropout=dropout,
        use_temperature=temperature,
        alpha_value=alpha_value,
        learnable_alpha=learnable_alpha,
        use_residual=use_residual,
        use_go_bias=use_go_bias,
        freeze_go_embedding=freeze_go_embedding
    ).to(device)

    if temperature:
        print(f"Using learnable temperature scaling (initial: {combine_model.temperature.item():.2f})")

    # 损失函数
    if loss == 'focal':
        criterion = FocalLoss(alpha=0.2, gamma=2)
        print("Using Focal Loss")
    elif loss == 'asymmetric':
        criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=0, clip=0.05, eps=1e-8)
        print("Using Asymmetric Loss")
    elif loss == 'combined':
        criterion = CombinedFocalBCELoss(alpha=0.2, gamma=2, focal_weight=0.7)
        print("Using Combined Focal+BCE Loss")
    else:
        criterion = lambda pred, target: F.binary_cross_entropy(pred, target)
        print("Using Binary Cross Entropy Loss")

    # Early stopping
    best_val = float("inf")
    best_epoch = -1
    wait = 0
    patience = 8
    min_delta = 1e-6

    print(combine_model)
    train_labels = train_labels.detach().cpu().numpy()
    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()

    # 优化器
    optimizer = torch.optim.Adam(combine_model.parameters(), lr=lr)

    # 学习率调度器
    if scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    elif scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
    elif scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    else:
        lr_scheduler = None

    # 训练循环
    best_loss = 100000.00
    results_log = []

    for epoch in range(epoch_num):
        combine_model.train()
        train_loss = 0
        train_steps = int(math.ceil(len(train_labels) / TRAIN_BS))

        with ck.progressbar(length=train_steps, show_pos=True) as bar:
            for batch in train_dataloader:
                bar.update(1)
                batch_labels = batch['labels'].to(device)
                batch_features = batch['esm2_embeddings'].to(device)
                train_output = combine_model(batch_features).to(device)
                loss = criterion(train_output, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.detach().item()

        train_loss /= train_steps

        # 学习率调度
        if scheduler == 'plateau' and lr_scheduler is not None:
            pass  # 在验证后更新
        elif lr_scheduler is not None:
            lr_scheduler.step()

        # 验证
        print('validation')
        combine_model.eval()
        with torch.no_grad():
            valid_steps = int(math.ceil(len(valid_labels) / EVAL_BS))
            valid_loss = 0
            preds = []

            with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                for batch in valid_dataloader:
                    bar.update(1)
                    batch_labels = batch['labels'].to(device)
                    batch_features = batch['esm2_embeddings'].to(device)
                    valid_output = combine_model(batch_features)
                    batch_loss = criterion(valid_output, batch_labels)
                    valid_loss += batch_loss.detach().item()
                    preds = np.append(preds, valid_output.detach().cpu().numpy())

            valid_loss /= valid_steps

            # Plateau调度器在这里更新
            if scheduler == 'plateau' and lr_scheduler is not None:
                lr_scheduler.step(valid_loss)

            roc_auc = compute_roc(valid_labels, preds)
            current_lr = optimizer.param_groups[0]['lr']

            print(f'Epoch {epoch}: Loss - {train_loss:.4f} Valid loss - {valid_loss:.4f}, AUC - {roc_auc:.4f}, LR - {current_lr:.6f}')

            # 记录
            results_log.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'valid_auc': roc_auc,
                'lr': current_lr
            })

        # 保存最佳模型
        if valid_loss < best_loss - min_delta:
            best_loss = valid_loss
            best_epoch = epoch
            print(f'✓ New best validation loss: {best_loss:.4f}')
            wait = 0
            torch.save(combine_model.state_dict(), model_save_file_path)
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # 测试
    print('Loading the best model')
    combine_model.load_state_dict(torch.load(model_save_file_path))
    combine_model.eval()

    with torch.no_grad():
        test_loss = 0
        test_steps = int(math.ceil(len(test_labels) / EVAL_BS))
        preds = []

        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for batch in test_dataloader:
                bar.update(1)
                batch_labels = batch['labels'].to(device)
                batch_features = batch['esm2_embeddings'].to(device)
                test_output = combine_model(batch_features).to(device)
                batch_loss = criterion(test_output, batch_labels)
                test_loss += batch_loss.detach().item()
                preds.append(test_output.cpu().numpy())

        test_loss /= test_steps
        preds = np.concatenate(preds)
        roc_auc = compute_roc(test_labels, preds)

        print(f'\n{"="*80}')
        print(f'Valid Loss - {valid_loss:.4f}, Test Loss - {test_loss:.4f}, Test AUC - {roc_auc:.4f}')

        if temperature:
            print(f'Learned temperature: {combine_model.temperature.item():.4f}')

        if learnable_alpha:
            alpha_val = combine_model.alpha.item() if isinstance(combine_model.alpha, nn.Parameter) else combine_model.alpha
            print(f'Learned alpha: {alpha_val:.4f}')

        print(f'Best epoch: {best_epoch}')
        print(f'{"="*80}\n')

    # 保存结果日志
    results_df = pd.DataFrame(results_log)
    results_df.to_csv(f'../data/model_checkpoint/training_log_{exp_id}.csv', index=False)

    # 预测
    preds = list(preds)
    with Pool(32) as p:
        preds = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict), preds)
    test_df['preds'] = preds
    test_df.to_pickle(f'{data_root}/{aspect}/predictions_{exp_id}.pkl')

    print(f"Results saved with experiment ID: {exp_id}")


# 消融实验版本的模型 (支持更多控制参数)
class Combine_Transformer_Ablation(Combine_Transformer):
    def __init__(self, num_heads, go_context, embedding_vector, device, num_layers=1, dropout=0.1,
                 use_temperature=False, alpha_value=None, learnable_alpha=True,
                 use_residual=True, use_go_bias=True, freeze_go_embedding=False):

        # 先不调用super().__init__，手动初始化
        nn.Module.__init__(self)

        self.heads = num_heads
        self.num_layers = num_layers
        self.use_temperature = use_temperature
        self.use_residual = use_residual
        self.use_go_bias = use_go_bias
        self.go_context = go_context.to(device)

        # GO嵌入 - 可选冻结
        if freeze_go_embedding:
            self.register_buffer('go_embedding_vector', embedding_vector.clone().detach())
        else:
            self.go_embedding_vector = nn.Parameter(embedding_vector.clone().detach())

        hidden_dim = self.go_context.shape[1]

        # fc1 和 fc1_5 - 可选残差
        self.fc1 = mlpblock(2560, hidden_dim, layer_norm=False, dropout=dropout).to(device)
        if use_residual:
            self.fc1_5 = Residual(mlpblock(hidden_dim, hidden_dim, layer_norm=False, dropout=dropout)).to(device)
        else:
            self.fc1_5 = mlpblock(hidden_dim, hidden_dim, layer_norm=False, dropout=dropout).to(device)

        # fc2 - 可选残差
        if use_residual:
            self.fc2 = Residual(mlpblock(hidden_dim, hidden_dim, layer_norm=False, dropout=dropout)).to(device)
        else:
            self.fc2 = mlpblock(hidden_dim, hidden_dim, layer_norm=False, dropout=dropout).to(device)

        # GO bias - 可选
        if use_go_bias:
            self.go_bias = nn.Parameter(torch.zeros(self.go_context.shape[0])).to(device)
        else:
            self.register_buffer('go_bias', torch.zeros(self.go_context.shape[0]).to(device))

        self.training = True

        # Alpha - 可学习或固定
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_value or 0.5))
        else:
            self.alpha = alpha_value if alpha_value is not None else 0.5

        # Temperature
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        else:
            self.temperature = 1.0

        # 交叉注意力层
        if num_layers == 1:
            self.cross_attention_fusion = crossattentionfusion(dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        else:
            self.attention_layers = nn.ModuleList([
                crossattentionfusion(dim=hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ])


if __name__ == '__main__':
    main()
