import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt

def run():
    for root, dirs, files in os.walk('data/pnl/'):
        if root != 'data/pnl/':
            count = 0
            for file in files:
                curr_df = pd.read_csv(root + '/' + file, index_col=0)
                count += 1
                try:
                    all_df = pd.concat([all_df, curr_df], sort=False)
                except:
                    all_df = curr_df
            all_df = all_df.groupby(all_df.index)
            ave_df = all_df.mean().round(3)
            std_df = all_df.std().round(3)
            if not os.path.exists('data/pnl/aggregate'):
                os.mkdir('data/aggregate')
            curr_dir = root.split('/')[2]

            mat_plot(ave_df, f'Mean Plot - {curr_dir.capitalize()}')
            mat_plot(std_df, f'STD Plot - {curr_dir.capitalize()}')
            plt.show()
            ave_df.to_csv(f'data/aggregate/{curr_dir}_mean.csv')
            std_df.to_csv(f'data/aggregate/{curr_dir}_std.csv')


def mat_plot(df, title):
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title(f'{title}', fontsize=16)

if __name__ == '__main__':
    run()