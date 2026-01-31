#!/usr/bin/env python3
"""
消融实验结果可视化脚本
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AblationResultsVisualizer:
    def __init__(self, results_file, output_dir='ablation_plots'):
        self.results_file = results_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 加载结果
        self.df = pd.read_csv(results_file)
        print(f"已加载 {len(self.df)} 条实验记录")

    def plot_component_ablation(self, metric='aupr', figsize=(12, 6)):
        """
        绘制组件消融对比图
        显示移除每个组件后的性能变化
        """
        # 按实验ID分组统计
        grouped = self.df.groupby('exp_id')[metric].agg(['mean', 'std']).reset_index()
        grouped = grouped.sort_values('mean', ascending=False)

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制条形图
        x = np.arange(len(grouped))
        bars = ax.bar(x, grouped['mean'], yerr=grouped['std'],
                     capsize=5, alpha=0.7, edgecolor='black')

        # 为最佳结果标色
        best_idx = grouped['mean'].idxmax()
        bars[best_idx].set_color('green')
        bars[best_idx].set_alpha(0.9)

        # 设置标签
        ax.set_xlabel('实验配置', fontsize=12)
        ax.set_ylabel(f'{metric.upper()}', fontsize=12)
        ax.set_title(f'消融实验结果对比 ({metric.upper()})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(grouped['exp_id'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 添加数值标签
        for i, (mean_val, std_val) in enumerate(zip(grouped['mean'], grouped['std'])):
            ax.text(i, mean_val + std_val + 0.005, f'{mean_val:.4f}',
                   ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # 保存图片
        save_path = os.path.join(self.output_dir, f'component_ablation_{metric}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()

    def plot_loss_function_comparison(self, metric='aupr'):
        """
        绘制不同损失函数的性能对比
        """
        # 筛选损失函数实验
        loss_df = self.df[self.df['exp_id'].str.contains('Exp-7', na=False)]

        if loss_df.empty:
            print("未找到损失函数对比实验 (Exp-7)")
            return

        grouped = loss_df.groupby('loss')[metric].agg(['mean', 'std']).reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(grouped))
        bars = ax.bar(x, grouped['mean'], yerr=grouped['std'],
                     capsize=5, alpha=0.7, edgecolor='black',
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])

        ax.set_xlabel('损失函数', fontsize=12)
        ax.set_ylabel(f'{metric.upper()}', fontsize=12)
        ax.set_title(f'损失函数性能对比 ({metric.upper()})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(grouped['loss'], rotation=0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 添加数值标签
        for i, (mean_val, std_val) in enumerate(zip(grouped['mean'], grouped['std'])):
            ax.text(i, mean_val + std_val + 0.005, f'{mean_val:.4f}±{std_val:.4f}',
                   ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'loss_function_comparison_{metric}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()

    def plot_temperature_effect(self):
        """
        绘制温度缩放的影响
        """
        # 筛选温度实验
        temp_df = self.df[self.df['exp_id'].str.contains('Exp-6', na=False)]

        if temp_df.empty:
            print("未找到温度缩放实验 (Exp-6)")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # AUPR对比
        grouped_aupr = temp_df.groupby('exp_id')['aupr'].agg(['mean', 'std']).reset_index()
        x = np.arange(len(grouped_aupr))
        ax1.bar(x, grouped_aupr['mean'], yerr=grouped_aupr['std'],
               capsize=5, alpha=0.7, edgecolor='black', color='steelblue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(grouped_aupr['exp_id'], rotation=45, ha='right')
        ax1.set_ylabel('AUPR', fontsize=12)
        ax1.set_title('温度缩放对AUPR的影响', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # AUROC对比
        grouped_auroc = temp_df.groupby('exp_id')['auroc'].agg(['mean', 'std']).reset_index()
        ax2.bar(x, grouped_auroc['mean'], yerr=grouped_auroc['std'],
               capsize=5, alpha=0.7, edgecolor='black', color='coral')
        ax2.set_xticks(x)
        ax2.set_xticklabels(grouped_auroc['exp_id'], rotation=45, ha='right')
        ax2.set_ylabel('AUROC', fontsize=12)
        ax2.set_title('温度缩放对AUROC的影响', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'temperature_scaling_effect.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()

    def plot_heatmap(self, metrics=['aupr', 'auroc']):
        """
        绘制实验结果热力图
        """
        # 准备数据
        pivot_data = []
        for metric in metrics:
            grouped = self.df.groupby('exp_id')[metric].mean().reset_index()
            pivot_data.append(grouped.set_index('exp_id')[metric])

        heatmap_df = pd.DataFrame(pivot_data, index=[m.upper() for m in metrics]).T

        fig, ax = plt.subplots(figsize=(8, len(self.df['exp_id'].unique()) * 0.5))
        sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='YlGnBu',
                   cbar_kws={'label': '性能指标'}, ax=ax)
        ax.set_title('消融实验性能热力图', fontsize=14, fontweight='bold')
        ax.set_xlabel('评估指标', fontsize=12)
        ax.set_ylabel('实验配置', fontsize=12)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'performance_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()

    def plot_radar_chart(self, exp_ids=None, metrics=['aupr', 'auroc', 'fmax']):
        """
        绘制雷达图对比多个实验配置
        """
        if exp_ids is None:
            # 选择AUPR最高的5个实验
            top_exps = self.df.groupby('exp_id')['aupr'].mean().nlargest(5).index.tolist()
            exp_ids = top_exps

        # 筛选数据
        plot_df = self.df[self.df['exp_id'].isin(exp_ids)]

        # 准备雷达图数据
        categories = [m.upper() for m in metrics]
        N = len(categories)

        # 计算角度
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        # 为每个实验绘制雷达图
        for exp_id in exp_ids:
            exp_data = plot_df[plot_df['exp_id'] == exp_id]
            values = [exp_data[m].mean() for m in metrics]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=exp_id)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('实验配置多指标对比 (雷达图)', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'radar_chart_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()

    def plot_performance_delta(self, baseline_exp='Exp-1A', metric='aupr'):
        """
        绘制相对于基线的性能变化
        """
        # 获取基线性能
        baseline_perf = self.df[self.df['exp_id'] == baseline_exp][metric].mean()

        if pd.isna(baseline_perf):
            print(f"未找到基线实验: {baseline_exp}")
            return

        # 计算性能变化
        grouped = self.df.groupby('exp_id')[metric].mean().reset_index()
        grouped['delta'] = ((grouped[metric] - baseline_perf) / baseline_perf) * 100
        grouped = grouped[grouped['exp_id'] != baseline_exp]  # 移除基线本身
        grouped = grouped.sort_values('delta', ascending=True)

        # 绘制
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ['green' if x > 0 else 'red' for x in grouped['delta']]
        bars = ax.barh(range(len(grouped)), grouped['delta'], color=colors, alpha=0.7, edgecolor='black')

        ax.set_yticks(range(len(grouped)))
        ax.set_yticklabels(grouped['exp_id'])
        ax.set_xlabel(f'相对性能变化 (%) [基线: {baseline_exp}]', fontsize=12)
        ax.set_title(f'消融实验性能变化 ({metric.upper()})', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)

        # 添加数值标签
        for i, (delta_val, perf_val) in enumerate(zip(grouped['delta'], grouped[metric])):
            label = f'{delta_val:+.2f}% ({perf_val:.4f})'
            x_pos = delta_val + (0.5 if delta_val > 0 else -0.5)
            ax.text(x_pos, i, label, va='center', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'performance_delta_{metric}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()

    def generate_all_plots(self):
        """生成所有可视化图表"""
        print("\n开始生成可视化图表...")

        # 1. 组件消融对比
        if 'aupr' in self.df.columns:
            self.plot_component_ablation(metric='aupr')
            self.plot_component_ablation(metric='auroc')

        # 2. 损失函数对比
        if 'loss' in self.df.columns:
            self.plot_loss_function_comparison(metric='aupr')

        # 3. 温度缩放效果
        self.plot_temperature_effect()

        # 4. 热力图
        available_metrics = [m for m in ['aupr', 'auroc', 'fmax'] if m in self.df.columns]
        if available_metrics:
            self.plot_heatmap(metrics=available_metrics)

        # 5. 雷达图
        if len(available_metrics) >= 2:
            self.plot_radar_chart(metrics=available_metrics[:3])

        # 6. 性能变化图
        if 'aupr' in self.df.columns and 'Exp-1A' in self.df['exp_id'].values:
            self.plot_performance_delta(baseline_exp='Exp-1A', metric='aupr')

        print(f"\n所有图表已保存到: {self.output_dir}")

    def generate_latex_table(self):
        """
        生成LaTeX格式的实验结果表格
        """
        # 按实验ID分组
        grouped = self.df.groupby('exp_id').agg({
            'aupr': ['mean', 'std'],
            'auroc': ['mean', 'std'],
        }).round(4)

        # 计算相对于基线的变化
        if 'Exp-1A' in grouped.index:
            baseline_aupr = grouped.loc['Exp-1A', ('aupr', 'mean')]
            grouped['delta_aupr'] = ((grouped[('aupr', 'mean')] - baseline_aupr) / baseline_aupr * 100).round(2)
        else:
            grouped['delta_aupr'] = 0.0

        # 生成LaTeX代码
        latex_lines = []
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{消融实验结果}")
        latex_lines.append("\\label{tab:ablation}")
        latex_lines.append("\\begin{tabular}{lcccc}")
        latex_lines.append("\\toprule")
        latex_lines.append("Model Variant & AUPR $\\uparrow$ & AUROC $\\uparrow$ & $\\Delta$AUPR (\\%) \\\\")
        latex_lines.append("\\midrule")

        for exp_id in grouped.index:
            row = grouped.loc[exp_id]
            aupr_mean = row[('aupr', 'mean')]
            aupr_std = row[('aupr', 'std')]
            auroc_mean = row[('auroc', 'mean')]
            auroc_std = row[('auroc', 'std')]
            delta = row['delta_aupr']

            line = f"{exp_id} & {aupr_mean:.4f}$\\pm${aupr_std:.4f} & {auroc_mean:.4f}$\\pm${auroc_std:.4f} & {delta:+.2f} \\\\"
            latex_lines.append(line)

        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")

        latex_code = "\n".join(latex_lines)

        # 保存到文件
        latex_file = os.path.join(self.output_dir, 'ablation_table.tex')
        with open(latex_file, 'w') as f:
            f.write(latex_code)

        print(f"\nLaTeX表格已保存到: {latex_file}")
        print("\n" + "="*80)
        print("LaTeX代码预览:")
        print("="*80)
        print(latex_code)
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='可视化消融实验结果')
    parser.add_argument('--results-file', required=True,
                       help='实验结果CSV文件路径')
    parser.add_argument('--output-dir', default='ablation_plots',
                       help='输出目录')
    parser.add_argument('--latex', action='store_true',
                       help='生成LaTeX表格')

    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"错误: 未找到结果文件 {args.results_file}")
        return

    # 创建可视化器
    visualizer = AblationResultsVisualizer(args.results_file, args.output_dir)

    # 生成所有图表
    visualizer.generate_all_plots()

    # 生成LaTeX表格
    if args.latex:
        visualizer.generate_latex_table()


if __name__ == '__main__':
    main()
