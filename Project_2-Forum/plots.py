import pandas as pd
from pyecharts.charts import Line, Scatter, Page
import pyecharts.options as opts
import os
import re
import numpy as np

page = Page()


def plot_lines(pnls):
    """
    Plot the line chart for every ret_type
    :param pnls: pd.DataFrame that stores the averaged return for every return type
    :return: None
    """
    csi300 = pd.read_csv('data/target_list/csi300_prices.csv', index_col=0, parse_dates=True)
    csi300_close = csi300['Price']
    csi300_close = csi300_close.apply(lambda row: float(''.join(re.findall('(\d)+,([\d.]+)', row)[0])))
    csi300_close = (csi300_close - csi300_close.shift(-1)) / csi300_close.shift(-1)
    csi300_close.name = 'CSI300'

    csi300_open = csi300['Open']
    csi300_open = csi300_open.apply(lambda row: float(''.join(re.findall('(\d)+,([\d.]+)', row)[0])))
    csi300_open = (csi300_open - csi300_open.shift(-1)) / csi300_open.shift(-1)
    csi300_open.name = 'CSI300_OPEN'

    pnls = pd.concat([pnls, csi300_close, csi300_open], axis=1).dropna(axis=0)
    pnls = pnls.round(4)

    """This is to process the returns for statistics calculation"""
    excess_daily = pd.DataFrame(index=pnl.index)
    for col in pnls.columns:
        if col.find('CSI300'):
            curr_ret = pnls.loc[:, col]
            if col == 'omo_ret':
                excess_ret = curr_ret - pnls.loc[:, 'CSI300_OPEN']
            else:
                excess_ret = curr_ret - pnls.loc[:, 'CSI300']
        else:
            excess_ret = pnls.loc[:, col]
        excess_ret.name = col
        excess_daily = pd.concat([excess_daily, excess_ret], axis=1)
    excess_daily.to_csv('data/interim/daily_pnls.csv')

    """This is to process the returns for drawing the cumulative return plot"""
    cumulative_pnl = pd.DataFrame(index=pnl.index)
    for col in pnls.columns:
        if (col != 'CSI300') and (col != 'CSI300_OPEN'):
            curr_ret = pnls.loc[:, col]
            if col == 'omo_ret':
                excess_ret = curr_ret - pnls.loc[:, 'CSI300_OPEN']
            else:
                excess_ret = curr_ret - pnls.loc[:, 'CSI300']
        else:
            excess_ret = pnls.loc[:, col]
        excess_ret = excess_ret.apply(lambda x: x + 1)
        excess_ret = excess_ret.cumprod()
        excess_ret = excess_ret - 1
        if (col != 'CSI300') and (col != 'CSI300_OPEN'):
            excess_ret.name = 'excess_' + col
            curr_ret = curr_ret.apply(lambda x: x + 1)
            curr_ret = curr_ret.cumprod() - 1
            cumulative_pnl = pd.concat([cumulative_pnl, curr_ret, excess_ret], axis=1)
        else:
            excess_ret.name = col
            cumulative_pnl = pd.concat([cumulative_pnl, excess_ret], axis=1)
    cumulative_pnl = cumulative_pnl.round(3)

    line = Line(init_opts=opts.InitOpts(width="1200px", height="600px"))
    line.add_xaxis(
        xaxis_data=[date.strftime('%Y-%m-%d') for date in
                    pnls.index])  # input of x-axis has been string format
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


def plot_scatter(pnls):
    """
    This is to plot scatter plot for detecting the outliers
    :param pnls: list of tuples that contains the (ret_type, pnl: pd.DataFrame)
    :return: None
    """
    scatter = Scatter(init_opts=opts.InitOpts(width="1600px", height="1000px"))
    for ret_type, pnl in pnls:
        pnl_aggregate = compute_mean_std(pnl)
        scatter.add_xaxis(xaxis_data=pnl_aggregate.loc['mean', :].values.tolist())
        scatter.add_yaxis(
            series_name=f'{ret_type.upper()}',
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
            title_opts=opts.TitleOpts(title=f'Scatter of {ret_type.capitalize()}'.upper(), pos_left='0%'),
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


def compute_daily_return(pnl, col_name):
    """
    Compute the nonzero average daily return in row (every timestamp's mean)
    :param pnl: pd.DataFrame, every stocks returns under a ret_type
    :param col_name: ret_type
    :return: pd.Series, a averaged mean series
    """
    means = []
    for row in pnl.index:
        curr_row = pnl.loc[row, :]
        mean = np.mean(curr_row.iloc[np.nonzero(curr_row.values)])
        means.append(mean)
    return pd.Series(means, index=pnl.index, name=col_name)


if __name__ == '__main__':
    files = os.listdir('data/interim')
    pnl_files = [file for file in files if re.match('individual[\w_.]+', file)]
    pnls = []

    for pnl_file in pnl_files:
        pnl = pd.read_csv(f'data/interim/{pnl_file}', index_col=0, parse_dates=True)
        pnl = pnl.drop('Total', axis=1)
        try:
            curr_series = compute_daily_return(pnl, re.findall('pnl_([\w_]+).csv', pnl_file)[0])
            all_total = pd.concat([all_total, curr_series], axis=1)
        except:
            all_total = compute_daily_return(pnl, re.findall('pnl_([\w_]+).csv', pnl_file)[0])
        pnls.append((re.findall('pnl_([\w_]+).csv', pnl_file)[0], pnl))

    plot_lines(all_total.dropna())
    plot_scatter(pnls)
    page.render(path='figs/comparison_plots.html')