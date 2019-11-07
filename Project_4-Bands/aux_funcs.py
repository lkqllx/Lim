from pyecharts.charts import *
import pyecharts.options as opts
import pandas as pd
import datetime as dt
import jqdatasdk as jq
import os


def plot_bolling_bandas():
    if not os.path.exists('data/csi.csv'):
        csi = jq.get_price(jq.normalize_code('000300.XSHG'), start_date='2006-11-06', end_date='2019-11-06')
        csi.to_csv('data/csi.csv')
    else:
        csi = pd.read_csv('data/csi.csv', index_col=0, parse_dates=True)
    # csi = csi.loc[:, ['open', 'high', 'low', 'close', 'volume', 'money']]
    data = csi.iloc[:, 0:4].values.tolist()
    dates = csi.index.strftime('%Y-%m-%d')
    dates = [date for date in dates.values.tolist()]
    kline = (
        Kline(init_opts=opts.InitOpts(width='1600px', height='1200'))
        .add_xaxis(dates)
        .add_yaxis("Kline of CSI300", data)
        # .set_global_opts(
        # xaxis_opts=opts.AxisOpts(is_scale=True),
        # yaxis_opts=opts.AxisOpts(
        #     is_scale=True,
        #     splitarea_opts=opts.SplitAreaOpts(
        #         is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
        #     ),
        # ),
        # datazoom_opts=[opts.DataZoomOpts(pos_bottom="-2%")],
        # title_opts=opts.TitleOpts(title="Double Bollinger Bands"),
        # legend_opts=opts.LegendOpts(pos_top='5%')
        # )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="CSI300 Kline",
            ),
            xaxis_opts=opts.AxisOpts(type_="category"),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            legend_opts=opts.LegendOpts(
                is_show=False, pos_bottom='10%', pos_left="center"
            ),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=False,
                    type_="inside",
                    xaxis_index=[0, 1],
                    range_start=0,
                    range_end=100,
                ),
                opts.DataZoomOpts(
                    is_show=True,
                    xaxis_index=[0, 1],
                    type_="slider",
                    pos_top="90%",
                    range_start=0,
                    range_end=100,
                ),
            ],
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
                background_color="rgba(245, 245, 245, 0.8)",
                border_width=1,
                border_color="#ccc",
                textstyle_opts=opts.TextStyleOpts(color="#000"),
            ),
            visualmap_opts=opts.VisualMapOpts(
                dimension=2,
                is_piecewise=True,
                pieces=[
                    {"value": 1, "color": "#ec0000"},
                    {"value": -1, "color": "#00da3c"},
                ],
            ),
        )
    )
    moving_10_ave = csi.close.rolling(10, min_periods=1).mean()
    std = csi.close.rolling(10, min_periods=1).std().fillna(0)

    line = (
        Line()
        .add_xaxis(dates)
        .add_yaxis('10 Days Moving Average', moving_10_ave.round(1).values.tolist())
        .add_yaxis('Upper 1 STD', (moving_10_ave + std).round(1).values.tolist())
        .add_yaxis('Upper 2 STD', (moving_10_ave + 2 * std).round(1).values.tolist())
        .add_yaxis('Lower 1 STD', (moving_10_ave - std).round(1).values.tolist())
        .add_yaxis('Lower 2 STD', (moving_10_ave - 2 * std).round(1).values.tolist())
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )
    kline.overlap(line)

    bar = (
        Bar()
            .add_xaxis(xaxis_data=dates)
            .add_yaxis(
            series_name="Volume",
            yaxis_data=[
                [i, csi.iloc[i, 4], 1 if data[i][1] > data[i][0] else -1]
                for i in range(len(data))
            ],
            xaxis_index=1,
            yaxis_index=1,
            label_opts=opts.LabelOpts(is_show=False),
        )
            .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                is_scale=True,
                grid_index=1,
                boundary_gap=False,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(is_show=False),
                split_number=20,
                min_="dataMin",
                max_="dataMax",
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=1,
                is_scale=True,
                split_number=2,
                axislabel_opts=opts.LabelOpts(is_show=False),
                axisline_opts=opts.AxisLineOpts(is_show=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )

    grid_chart = Grid()
    grid_chart.add(
        kline,
        grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", height="50%"),
    )
    grid_chart.add(
        bar,
        grid_opts=opts.GridOpts(
            pos_left="10%", pos_right="8%", pos_top="70%", height="16%"
        ),
    )

    grid_chart.render()


if __name__ == '__main__':
    jq.auth('18810906018', '906018')
    plot_bolling_bandas()