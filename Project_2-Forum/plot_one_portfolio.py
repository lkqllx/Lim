import pandas as pd
from pyecharts.charts import Line, Scatter, Page
import pyecharts.options as opts
import os
import re
import numpy as np
from progress.bar import Bar


page = Page()

def plot_aggregate_excess_return():
    files = os.listdir('data/interim/aggregate_protfolios/')
    pnl_files = [file for file in files if re.match('[\w_.]+transac.csv', file)]
    pnl_files = sorted(pnl_files, key=lambda x: int(re.findall('cmc(\d+)_.+', x)[0]))
    for pnl_file in pnl_files:
        pnl = pd.read_csv('data/interim/aggregate_protfolios/' + pnl_file, index_col=0, parse_dates=True).dropna()
        pnl_mean = pnl.mean(axis=1)
        interval = re.findall('cmc(\d+)_.+', pnl_file)[0]
        pnl_mean.name = f'interval {interval}'
        try:
            all_mean = pd.concat([all_mean, pnl_mean], axis=1)
        except:
            all_mean = pnl_mean

    cumulative_pnl = pd.DataFrame(index=all_mean.index)
    for col in all_mean.columns:
        curr_ret = all_mean.loc[:, col]
        curr_ret = curr_ret.apply(lambda x: x + 1)
        curr_ret = curr_ret.cumprod() - 1
        cumulative_pnl = pd.concat([cumulative_pnl, curr_ret], axis=1)
    cumulative_pnl = cumulative_pnl.round(3)
    draw_line_plot(cumulative_pnl)

# TODO -> Have not dealt with the ave_returns case after creating a lot of offets
def plot_lines(offest):
    """
    Plot the line chart for every ret_type
    :param pnls: pd.DataFrame that stores the averaged return for every return type
    :return: None
    """

    csi300 = pd.read_csv('data/target_list/csi300_prices.csv', index_col=0, parse_dates=True)
    csi300_close = csi300['Price']
    csi300_close = csi300_close.apply(lambda row: float(''.join(re.findall('(\d)+,([\d.]+)', row)[0])))
    csi300_close_ret = (csi300_close - csi300_close.shift(-1)) / csi300_close.shift(-1)
    csi300_close_ret.name = f'CSI300_cmc1'



    files = os.listdir('data/interim/ave_returns/')
    pnl_files = [file for file in files if re.match(f'cmc[\w_.]+ret_{offest}.csv', file)]
    pnl_files = sorted(pnl_files, key=lambda x: int(re.findall(f'cmc(\d+)_.+{offest}.csv', x)[0]))
    for idx, pnl_file in enumerate(pnl_files):
        pnl = pd.read_csv(f'data/interim/ave_returns/{pnl_file}', index_col=0, parse_dates=True)
        interval = re.findall('cmc(\d+)_.+', pnl_file)[0]
        start = re.findall('_(\d+).csv', pnl_file)[0]
        pnl.columns = [f'interval {interval}_{start}', f'interval {interval}_{start}_transaction']
        if idx == 0:
            pnls = pnl
            continue
        pnls = pd.concat([pnls, pnl], axis=1)

    pnls = pd.concat([pnls, csi300_close_ret], axis=1).dropna(axis=0)
    pnls.to_csv('data/interim/daily_pnls.csv')

    # excess_pnls = pnls.sub(pnls['CSI300_cmc1'], axis=0)
    excess_pnls = pnls.drop('CSI300_cmc1', axis=1)
    excess_pnls = excess_pnls.round(4)
    excess_pnls.to_csv('data/interim/excess_daily_pnls.csv')

    """This is to process the returns for drawing the cumulative return plot"""
    cumulative_pnl = pd.DataFrame(index=excess_pnls.index)
    for col in excess_pnls.columns:
        if not re.match('CSI300.+', col):
            curr_ret = excess_pnls.loc[:, col]
            curr_ret = curr_ret.apply(lambda x: x + 1)
            curr_ret = curr_ret.cumprod() - 1
            cumulative_pnl = pd.concat([cumulative_pnl, curr_ret], axis=1)
    cumulative_pnl = cumulative_pnl.round(3)
    draw_line_plot(cumulative_pnl)

def draw_line_plot(cumulative_pnl):
    line = Line(init_opts=opts.InitOpts(width="1200px", height="600px"))
    line.add_xaxis(
        xaxis_data=[date.strftime('%Y-%m-%d') for date in
                    cumulative_pnl.index])  # input of x-axis has been string format
    for col in cumulative_pnl.columns:
        line.add_yaxis(y_axis=cumulative_pnl.loc[:, col].values.tolist(),
                       series_name=col.upper(),
                       is_smooth=True,
                       label_opts=opts.LabelOpts(is_show=False),
                       linestyle_opts=opts.LineStyleOpts(width=2)
                       )
    line.set_global_opts(
        datazoom_opts=opts.DataZoomOpts(),
        legend_opts=opts.LegendOpts(pos_top="5%", pos_right='70%', pos_left='10%'),
        title_opts=opts.TitleOpts(title='Total Returns Comparison'.upper(), pos_left='0%'),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross", is_show=True),
        xaxis_opts=opts.AxisOpts(boundary_gap=False, max_interval=5),
        yaxis_opts=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(formatter="{value}"),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        )
    )
    line.set_series_opts(
        markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='max', name='Max'),
                                                opts.MarkPointItem(type_='min', name='Min')]),
    )
    page.add(line)

def plot_scatter(offset):
    """
    This is to plot scatter plot for detecting the outliers
    :param pnls: list of tuples that contains the (ret_type, pnl: pd.DataFrame)
    :return: None
    """
    files = os.listdir('data/interim/one_portfolio_individual/')
    pnl_files = [file for file in files if re.match(f'individual[\w_.]+ret_{offset}', file)]
    pnl_files = sorted(pnl_files, key=lambda x: int(re.findall('.+cmc(\d+)_.+', x)[0]))

    scatter = Scatter(init_opts=opts.InitOpts(width="1600px", height="1000px"))
    for pnl_file in pnl_files:
        interval = re.findall(f'individual[\w_.]+ret_{offset}', pnl_file)[0]
        pnl = pd.read_csv(f'data/interim/one_portfolio_individual/{pnl_file}', index_col=0)
        pnl_aggregate = compute_mean_std(pnl)
        scatter.add_xaxis(xaxis_data=pnl_aggregate.loc['mean', :].values.tolist())
        scatter.add_yaxis(
            series_name=f'{interval.upper()}',
            y_axis=pnl_aggregate.loc['std', :].values.tolist(),
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
        )
        scatter.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                name='Mean',
                type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)
            ),
            yaxis_opts=opts.AxisOpts(
                name='Std',
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            title_opts=opts.TitleOpts(title=f'Scatter of {interval.capitalize()}'.upper(), pos_left='0%'),
            tooltip_opts=opts.TooltipOpts(is_show=False),
        )
        page.add(scatter)

def compute_mean_std(pnl):
    """
    Compute the nonzero mean and std in column (every stock's mean and std)
    :param pnl: pnl: pd.DataFrame, every stocks returns under a ret_type
    :return: pd.DataFrame, mean and std DataFrame for every stock
    """
    pnl_aggregate = pd.DataFrame(index=['mean', 'std'])
    for col in pnl.columns:
        curr_col = pnl.loc[:, col]
        mean = np.mean(curr_col.iloc[np.nonzero(curr_col.values)])
        std = np.std(curr_col.iloc[np.nonzero(curr_col.values)])
        pnl_aggregate = pd.concat([pnl_aggregate, pd.DataFrame([mean, std], columns=[col],
                                                               index=['mean', 'std'])], axis=1)
    return pnl_aggregate


if __name__ == '__main__':
    offset = 0
    plot_lines(offset)
    plot_scatter(offset)
    # plot_aggregate_excess_return()
    page.render(path='figs/one_portfolio_plots.html')