import pandas as pd
import pandas_datareader as web
from pyecharts.charts import Line, Scatter, Page
import pyecharts.options as opts
import os
import re

page = Page()

def plot_lines(pnls):
    shex = web.get_data_yahoo('000001.SS', start='2015-01-01', end='2019-07-31').Close
    shex.name = 'SSE Index'
    pnls = pd.concat([pnls, shex], axis=1).dropna(axis=0)
    pnls.loc[:, 'SSE Index'] =  pnls.loc[:, 'SSE Index'] / 100
    pnls = pnls.round(2)
    line = Line(init_opts=opts.InitOpts(width="1200px", height="600px"))
    line.add_xaxis(
        xaxis_data=[date.strftime('%Y-%m-%d') for date in
                    pnls.index])  # input of x-axis has been string format
    for col in pnls.columns:
        line.add_yaxis(y_axis=pnls.loc[:, col].values.tolist(),
                       series_name=col.capitalize(),
                       is_smooth=True,
                       label_opts=opts.LabelOpts(is_show=False),
                       linestyle_opts=opts.LineStyleOpts(width=2)
                       )
    line.set_global_opts(
        datazoom_opts=opts.DataZoomOpts(),
        legend_opts=opts.LegendOpts(pos_top="20%", pos_right='0%', pos_left='90%'),
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
    # return line

def plot_scatter(pnls):
    for ret_type, pnl in pnls:
        pnl_aggregate = pnl.agg(['mean', 'std'])

        scatter = Scatter(init_opts=opts.InitOpts(width="1600px", height="1000px"))
        scatter.add_xaxis(xaxis_data=pnl_aggregate.loc['mean', :].values.tolist())
        scatter.add_yaxis(
            series_name='',
            y_axis=pnl_aggregate.loc['std', :].values.tolist(),
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
        )
        # scatter.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}, {c}"))
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
        # scatters.append(scatter)
    # return scatters



if __name__ == '__main__':
    files = os.listdir('data/interim')
    pnl_files = [file for file in files if re.match('individual[\w_.]+', file)]
    pnls = []

    for pnl_file in pnl_files:
        pnl = pd.read_csv(f'data/interim/{pnl_file}', index_col=0, parse_dates=True)
        try:
            curr_series = pnl.loc[:, 'Total']
            curr_series.name = re.findall('pnl_([\w_]+).csv', pnl_file)[0]
            all_total = pd.concat([all_total, curr_series], axis=1)
        except:
            all_total = pnl.loc[:, 'Total']
            all_total.name = re.findall('pnl_([\w_]+).csv', pnl_file)[0]
        pnl = pnl.drop('Total', axis=1)
        pnls.append((re.findall('pnl_([\w_]+).csv', pnl_file)[0], pnl))

    plot_lines(all_total)
    plot_scatter(pnls)
    page.render(path='figs/comparison_plots.html')
    # Page(*[[line] + scatters]).render(path='figs/comparison_plots.html')