#!/usr/bin/env python3
"""
消融实验执行脚本
用于系统化地运行消融实验并记录结果
"""

import subprocess
import json
import os
from datetime import datetime
import pandas as pd
import argparse

class AblationExperimentRunner:
    def __init__(self, base_config, results_dir='ablation_results'):
        self.base_config = base_config
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # 初始化结果记录
        self.results = []
        self.results_file = os.path.join(results_dir, 'ablation_results.csv')

    def run_experiment(self, exp_id, description, config_override, seeds=[42]):
        """
        运行单个消融实验

        Args:
            exp_id: 实验ID (如 "Exp-1A")
            description: 实验描述
            config_override: 要覆盖的配置参数
            seeds: 随机种子列表
        """
        print(f"\n{'='*80}")
        print(f"运行实验: {exp_id} - {description}")
        print(f"{'='*80}")

        results_for_seeds = []

        for seed in seeds:
            print(f"\n--- Seed: {seed} ---")

            # 合并配置
            config = self.base_config.copy()
            config.update(config_override)
            config['seed'] = seed

            # 构建命令
            cmd = self._build_command(config)

            # 记录实验配置
            exp_log = {
                'exp_id': exp_id,
                'description': description,
                'seed': seed,
                'config': config,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # 保存实验配置
            config_file = os.path.join(self.results_dir, f'{exp_id}_seed{seed}_config.json')
            with open(config_file, 'w') as f:
                json.dump(exp_log, f, indent=2)

            print(f"命令: {' '.join(cmd)}")

            # 运行实验
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)

                # 解析结果 (需要根据实际输出格式调整)
                metrics = self._parse_output(result.stdout)

                # 记录结果
                exp_result = {
                    'exp_id': exp_id,
                    'description': description,
                    'seed': seed,
                    **config,
                    **metrics,
                    'success': result.returncode == 0
                }

                results_for_seeds.append(exp_result)
                self.results.append(exp_result)

                # 实时保存结果
                self._save_results()

                print(f"✓ 实验完成 - AUPR: {metrics.get('aupr', 'N/A')}, AUROC: {metrics.get('auroc', 'N/A')}")

            except Exception as e:
                print(f"✗ 实验失败: {str(e)}")
                exp_result = {
                    'exp_id': exp_id,
                    'description': description,
                    'seed': seed,
                    'success': False,
                    'error': str(e)
                }
                results_for_seeds.append(exp_result)
                self.results.append(exp_result)

        # 计算多种子的平均值和标准差
        if len(results_for_seeds) > 1:
            self._compute_statistics(exp_id, results_for_seeds)

        return results_for_seeds

    def _build_command(self, config):
        """根据配置构建命令行"""
        cmd = ['python', 'train_best_model.py']

        # 添加参数
        if 'aspect' in config:
            cmd.extend(['--aspect', config['aspect']])
        if 'loss' in config:
            cmd.extend(['--loss', config['loss']])
        if 'seed' in config:
            cmd.extend(['--seed', str(config['seed'])])
        if config.get('temperature', False):
            cmd.append('--temperature')
        if config.get('deepgo2', False):
            cmd.append('--deepgo2')
        if 'protein_embedding_path' in config:
            cmd.extend(['--protein_embedding_path', config['protein_embedding_path']])
        if 'go_embedding_path' in config:
            cmd.extend(['--go_embedding_path', config['go_embedding_path']])

        return cmd

    def _parse_output(self, output):
        """
        解析训练输出，提取评估指标
        需要根据实际的输出格式进行调整
        """
        metrics = {}

        # 示例: 假设输出包含 "Test AUC - 0.XXX"
        for line in output.split('\n'):
            if 'Test AUC' in line:
                try:
                    auroc = float(line.split('-')[-1].strip())
                    metrics['auroc'] = auroc
                except:
                    pass
            if 'AUPR' in line:
                try:
                    aupr = float(line.split(':')[-1].strip())
                    metrics['aupr'] = aupr
                except:
                    pass
            if 'Fmax' in line:
                try:
                    fmax = float(line.split(':')[-1].strip())
                    metrics['fmax'] = fmax
                except:
                    pass
            if 'Learned temperature' in line:
                try:
                    temp = float(line.split(':')[-1].strip())
                    metrics['final_temperature'] = temp
                except:
                    pass

        return metrics

    def _save_results(self):
        """保存结果到CSV"""
        df = pd.DataFrame(self.results)
        df.to_csv(self.results_file, index=False)
        print(f"结果已保存到: {self.results_file}")

    def _compute_statistics(self, exp_id, results_for_seeds):
        """计算多次运行的统计信息"""
        df = pd.DataFrame(results_for_seeds)

        if 'aupr' in df.columns:
            print(f"\n统计信息 ({exp_id}):")
            print(f"  AUPR: {df['aupr'].mean():.4f} ± {df['aupr'].std():.4f}")
        if 'auroc' in df.columns:
            print(f"  AUROC: {df['auroc'].mean():.4f} ± {df['auroc'].std():.4f}")
        if 'fmax' in df.columns:
            print(f"  Fmax: {df['fmax'].mean():.4f} ± {df['fmax'].std():.4f}")


def define_experiments():
    """定义所有消融实验"""
    experiments = []

    # ===== 实验1: 交叉注意力机制 =====
    experiments.append({
        'exp_id': 'Exp-1A',
        'description': '基线 - 单层交叉注意力',
        'config': {'aspect': 'mf', 'loss': 'focal'}
    })

    # 注意: Exp-1B需要修改代码设置alpha=0，这里仅作示例
    # experiments.append({
    #     'exp_id': 'Exp-1B',
    #     'description': '移除交叉注意力 (alpha=0)',
    #     'config': {'aspect': 'mf', 'loss': 'focal', 'alpha': 0.0}
    # })

    # ===== 实验2: 双路径融合策略 =====
    # 需要修改代码支持alpha参数

    # ===== 实验6: 温度缩放 =====
    experiments.append({
        'exp_id': 'Exp-6A',
        'description': '使用可学习温度缩放',
        'config': {'aspect': 'mf', 'loss': 'focal', 'temperature': True}
    })

    experiments.append({
        'exp_id': 'Exp-6B',
        'description': '无温度缩放',
        'config': {'aspect': 'mf', 'loss': 'focal', 'temperature': False}
    })

    # ===== 实验7: 损失函数对比 =====
    for loss_type in ['bce', 'focal', 'asymmetric', 'combined']:
        experiments.append({
            'exp_id': f'Exp-7-{loss_type.upper()}',
            'description': f'损失函数: {loss_type}',
            'config': {'aspect': 'mf', 'loss': loss_type}
        })

    return experiments


def main():
    parser = argparse.ArgumentParser(description='运行消融实验')
    parser.add_argument('--aspect', default='mf', choices=['mf', 'bp', 'cc'],
                       help='GO aspect')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                       help='随机种子列表 (如: --seeds 42 123 456)')
    parser.add_argument('--exp-ids', nargs='+', default=None,
                       help='要运行的实验ID列表 (如: --exp-ids Exp-1A Exp-6A). 不指定则运行所有实验')
    parser.add_argument('--results-dir', default='ablation_results',
                       help='结果保存目录')

    args = parser.parse_args()

    # 基础配置
    base_config = {
        'aspect': args.aspect,
    }

    # 初始化实验运行器
    runner = AblationExperimentRunner(base_config, results_dir=args.results_dir)

    # 定义实验
    all_experiments = define_experiments()

    # 筛选要运行的实验
    if args.exp_ids:
        experiments = [exp for exp in all_experiments if exp['exp_id'] in args.exp_ids]
        if not experiments:
            print(f"错误: 没有找到匹配的实验ID: {args.exp_ids}")
            print(f"可用的实验ID: {[exp['exp_id'] for exp in all_experiments]}")
            return
    else:
        experiments = all_experiments

    print(f"\n将运行 {len(experiments)} 个实验，每个实验使用 {len(args.seeds)} 个随机种子")
    print(f"总共: {len(experiments) * len(args.seeds)} 次训练\n")

    # 运行实验
    for exp in experiments:
        runner.run_experiment(
            exp_id=exp['exp_id'],
            description=exp['description'],
            config_override=exp['config'],
            seeds=args.seeds
        )

    print(f"\n{'='*80}")
    print("所有实验完成!")
    print(f"结果保存在: {args.results_dir}")
    print(f"{'='*80}\n")

    # 生成总结报告
    generate_summary_report(runner.results_file, args.results_dir)


def generate_summary_report(results_file, results_dir):
    """生成实验总结报告"""
    if not os.path.exists(results_file):
        print("未找到结果文件")
        return

    df = pd.read_csv(results_file)

    # 按实验ID分组，计算统计信息
    if 'aupr' in df.columns:
        summary = df.groupby('exp_id').agg({
            'aupr': ['mean', 'std', 'count'],
            'auroc': ['mean', 'std'],
        }).round(4)

        summary_file = os.path.join(results_dir, 'summary_report.csv')
        summary.to_csv(summary_file)

        print("\n实验总结:")
        print(summary)
        print(f"\n总结报告已保存到: {summary_file}")


if __name__ == '__main__':
    main()
