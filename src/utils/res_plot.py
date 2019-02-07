import os, sys
import argparse
import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

from env import DATA_PATH, ROOT_PATH

def plot_res_comparison(file):
    path = os.path.join(DATA_PATH, file)
    data = pd.read_csv(path, index_col=0)
    # amherst:0.918228364
    # blog:0.129837448473
    # youtube:0.1835647239
    data['LINE'] = 0.918228364
    data = data[['LINE', 'LINE-P', 'node2vec-P', 'DeepWalk-P', 'MANE']]
    fig, ax = plt.subplots(figsize=(8, 6))
    data.plot(style=['--', 'o-', 'x-', '+-', 's-'], ax=ax)
    # data.plot(kind='bar', ax=ax)
    # ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=4, prop={'size': 13})
    ax.legend(loc='lower right', prop={'size': 20})
    for tick in ax.get_xticklabels():
        tick.set_fontsize(20)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(20)
    plt.ylim((0.6, 1))
    plt.xlabel('Sub-graph scale upper bound', fontsize=20, fontweight='bold')
    # plt.ylabel('Macro-F1', fontsize=20, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=20, fontweight='bold')
    # plt.xticks(np.arange(50000, 450000, 50000), ('50k', '100k', '150k', '200k', '250k', '300k', '350k', '400k'), rotation=0)
    plt.tight_layout()
    plt.show()

def plot_acc_iter(file):
    path = os.path.join(DATA_PATH, file)
    image_dir = os.path.join(ROOT_PATH, 'images')
    data = pd.read_csv(path, header=[0, 1])
    sizes, tops = data.columns.levels[0].tolist(), data.columns.levels[1].tolist()

    for size in sizes:
        fig, ax = plt.subplots(figsize=(8, 6))
        size_num = size.replace('size', '')
        subdata = data.xs(size, axis=1, level=0)
        subdata.plot(style=['o-', 'x-', '+-', 's-'], ax=ax)
        for tick in ax.get_xticklabels():
            tick.set_fontsize(20)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(20)
        plt.xlabel('Epochs', fontsize=20, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=20, fontweight='bold')
        plt.ylim(0.7, 1)
        plt.legend(loc='lower right', prop={'size': 20})
        plt.tight_layout()
        # plt.show()
        image_path = os.path.join(image_dir, size + '.eps')
        plt.savefig(image_path)

    # for top in tops:
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     top_num = top.replace('top', '')
    #     subdata = data.xs(top, axis=1, level=1)
    #     subdata.plot(style=['o-', 'x-', '+-', 's-'], ax=ax)
    #     for tick in ax.get_xticklabels():
    #         tick.set_fontsize(20)
    #     for tick in ax.get_yticklabels():
    #         tick.set_fontsize(20)
    #     plt.xlabel('Epochs', fontsize=20, fontweight='bold')
    #     plt.ylabel('Accuracy', fontsize=20, fontweight='bold')
    #     plt.ylim(0.7, 1)
    #     plt.legend(loc='lower right', prop={'size': 20})
    #     plt.tight_layout()
    #     image_path = os.path.join(image_dir, top + '.eps')
    #     plt.savefig(image_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data', type=str, default="test")
    parser.add_argument('--func', type=str, default='plot_res_comparison')
    args = parser.parse_args()
    eval(args.func)(args.data)
