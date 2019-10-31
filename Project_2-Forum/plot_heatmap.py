from pyecharts.charts import Grid, HeatMap, Page, Tab
import pyecharts.options as opts
import pandas as pd
import os
import re
from progress.bar import Bar
import numpy as np
from itertools import product


def plot(path, method, direction, signal):
    csi300 = pd.read_csv('data/target_list/csi300_prices.csv', index_col=0, parse_dates=True)
    csi300_close = csi300['Price']
    csi300_close = csi300_close.apply(lambda row: float(''.join(re.findall('(\d)+,([\d.]+)', row)[0])))
    csi300_close_ret = (csi300_close - csi300_close.shift(-1)) / csi300_close.shift(-1)
    csi300_close_ret.name = f'CSI300_cmc1'

    all_pnls = {}
    with Bar('Computing PnLs', max=700) as bar:
        for root, dirs, files in os.walk(path):
            for dir_path in dirs:
                if re.match('Decile.+Counting.+', dir_path):
                    """For different folders"""
                    decile, count = re.findall('Decile ([\d.]+) - Counting ([\d]+)', dir_path)[0]
                    decile = float(decile)
                    for file in os.listdir(os.path.join(root, dir_path)):
                        if re.match('cmc.+csv', file):
                            """For different files"""
                            bar.next()
                            curr_df = pd.read_csv(os.path.join(root, dir_path, file),
                                                  index_col=0, parse_dates=True)

                            if direction == 'long':
                                curr_df = curr_df.iloc[:, 1]
                            else:
                                transac_cost =  curr_df.iloc[:, 0] - curr_df.iloc[:, 1]
                                curr_df = -curr_df.iloc[:, 0] - transac_cost

                            try:
                                if method == 'pnl':
                                    cum_pnl = comp_cum_pnl(curr_df, benchmark, direction)
                                else:
                                    cum_pnl = comp_cum_sharpe(curr_df, benchmark, direction)
                            except:
                                benchmark = pd.concat([curr_df, csi300_close_ret], axis=1).dropna()
                                benchmark = benchmark['CSI300_cmc1']
                                # benchmark.to_csv('data/bm.csv')
                                if method == 'pnl':
                                    cum_pnl = comp_cum_pnl(curr_df, benchmark, direction)
                                else:
                                    cum_pnl = comp_cum_sharpe(curr_df, benchmark, direction)

                            holding = file.split('_')[0]
                            try:
                                all_pnls[holding].append((int(decile * 2) - 1, int(count) - 1, cum_pnl))
                                # all_pnls[holding].append((int(decile) - 1, int(count) - 1, cum_pnl))
                            except:
                                all_pnls[holding] = [(int(decile * 2) - 1, int(count) - 1, cum_pnl)]
                                # all_pnls[holding] = [(int(decile) - 1, int(count) - 1, cum_pnl)]
    heatmap(all_pnls, method, direction, signal)


def heatmap(all_pnls, method, direction, signal):
    hms = []
    all_pnls = sorted(all_pnls.items(), key=lambda x: int(re.findall('cmc([\d]+)', x[0])[0]))
    for key, value in all_pnls:
        hm = HeatMap(init_opts=opts.InitOpts(page_title=f'', width='1200px', height='800px',))
        hm.add_xaxis(np.linspace(1, 20, 20))
        # hm.add_xaxis(list(range(1, 11)))
        hm.add_yaxis(f'{key.upper()}',
                     yaxis_data=list(range(1, 4)),
                     # yaxis_data=list(range(1, 11)),
                     value=value,
                    label_opts=opts.LabelOpts(is_show=True,
                                              color="#000",
                                              position="inside",)
                     )
        hm.set_global_opts(visualmap_opts=opts.VisualMapOpts(max_=2,
                                                             min_=-1,
                                                             split_number=5,
                                                             textstyle_opts=opts.TextStyleOpts(color="#000",
                                                                                               font_size=14),
                                                             is_calculable=True,
                                                             orient="horizontal"),
                           xaxis_opts=opts.AxisOpts(name='Decile of Signals'),
                           yaxis_opts=opts.AxisOpts(name='Look Back Period'),
                           legend_opts=opts.LegendOpts(is_show=True)
                           )
        hm.set_series_opts()
        hms.append(hm)

    # page = Page()
    # for idx, hm in enumerate(hms):
    #     if (idx != 0 ) and (idx % 2 == 0):
    #         grid = Grid(init_opts=opts.InitOpts(width='2400px', height='800px'))
    #         grid.add(hms[idx-1], grid_opts=opts.GridOpts(pos_left="60%"))
    #         grid.add(hms[idx], grid_opts=opts.GridOpts(pos_right="60%"))
    #         page.add(grid)
    # page.render(path='figs/heatmaps.html')
    tab = Tab(page_title=f'{method}_heatmaps_{signal}_{direction}'.upper())
    keys, values = list(zip(*all_pnls))
    for key, hm in zip(keys, hms):
        tab.add(hm, key)
    tab.render(path=f'figs/{method}_heatmaps_{signal}_{direction}.html')


def comp_cum_pnl(df, benchmark, direction):
    benchmark[0] = 0
    if direction =='long':
        df = df - benchmark
    else:
        df = df + benchmark
    df = df + 1
    cum_df = df.cumprod()
    return round(cum_df[-1] - 1, 3)

def comp_cum_sharpe(df, benchmark, direction):
    benchmark[0] = 0
    if direction =='long':
        df = df - benchmark
    else:
        df = df + benchmark
    std = df.std() * np.sqrt(255)
    df = df + 1
    df = df.apply(lambda x: x ** (255 / df.shape[0]))
    cum_df = df.cumprod()
    return round((cum_df[-1] - 1) / std, 3)


if __name__ == '__main__':
    methods = ['pnl', 'sharpe']
    signals = ['rank', 'change']
    directions = ['long', 'short']
    combs = list(product(methods, signals, directions))

    for method, signal, direction in combs:
        print(f'Doing {method} - {signal} - {direction}')
        path = f'data/params_top_{signal}'
        plot(path, method, direction, signal)