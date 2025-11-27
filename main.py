"""
主运行脚本
完整运行训练、评估和可视化流程
"""

import os
import torch
import json
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Multi30k En-De Translation Experiment')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['train', 'evaluate', 'visualize', 'all'],
                        help='运行模式')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cuda', 'cpu'], help='设备')
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"=" * 60)
    print(f"Multi30k English-German Translation Experiment")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"=" * 60)
    
    if args.mode in ['train', 'all']:
        print("\n>>> 开始训练...")
        from train import main as train_main
        train_main()
    
    if args.mode in ['evaluate', 'all']:
        print("\n>>> 开始评估...")
        from evaluate import main as eval_main
        eval_main()
    
    if args.mode in ['visualize', 'all']:
        print("\n>>> 生成可视化...")
        from visualize import generate_all_visualizations
        generate_all_visualizations()
    
    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
