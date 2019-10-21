"""
This file will back-test the forum strategy.

In v0.1, we will only test the performance from the trivial method - number of posts increment.
In v0.2, we will add the test of correlation between price and posts.

Note
    1.  We should exclude the IPO period for new stocks since we are not able to buy those stocks.
        Solution -> Skip a fixed period (20 days) after IPO to avoid trading in that period.

    2.  We cannot buy/sell those stocks if they have a 10% daily increase/decrease.
        Solution -> Skip the trading signal if its corresponding price change over 10%.

    3.  We need to check if the stock is tradable (exclude holiday, halted) or not.
        Solution -> The yahoo seems to be fine with this but Bloomberg will use the previous available data
        to replace all the unavailable prices. So we need to exclude those period if BBG is used.

    4.  The total comparable stocks should be greater 200 for crossing signals

    5.  We need to care about the number of posts by using historical average (like 20 days average)
        Reason -> If for stocks A and B,
                                    Stock A     Stock B
                    -> 2018-08-07     1          1000
                    -> 2018-08-08     2          1200
                    -> 2018-08-09     4          1500
        The trivial increment will be:
                                   Stock A     Stock B
                    -> 2018-08-08     2          1.2
                    -> 2018-08-09     2          1.25
        This trivial solution cannot show stock A attracts more attention than B
        Solution -> Instead of ranking the trivial increment matrix, we will further time it by log(ave(posts, days=20))
        which will give us a better sense about the market.
        For this scenario, we cannot say A is better than B so we time a log(ave_post)
        If we suppose ave(A, 20) = 2, ave(B, 20) = 1200, the updated matrix will be:
                                   Stock A     Stock B
                    -> 2018-08-08  2*log(2)    1.2*log(1000)
                    -> 2018-08-09  2*log(2)    1.25*log(1000)
"""
import matplotlib

matplotlib.rcParams['backend'] = 'TkAgg'
import pandas as pd
import datetime as dt
from progress.bar import Bar
import re
import os
import numpy as np
from snownlp import SnowNLP
from copy import deepcopy
from pyecharts.charts import Line
import pyecharts.options as opts
import matplotlib.pyplot as plt


class SelfSignal:
    """
    Trading signal for backtesting in terms of a single stock which means that
    the signal will be generated by comparing with its historical records
    """

    def __init__(self, ticker):
        self._ticker = ticker

    def create_signal(self):
        pass


class CrossSignal:
    """
    A trading signal matrix will be created by comparing among the universe. A row of signal is supposed to be
    the weightings of that day.

    The original posts matrix is
    Sample of stock_post_matrix:
                        Stock A     Stock B     Stock C
        -> 2018-08-07     17          10          10
        -> 2018-08-08     34          30          10
        -> 2018-08-09     30          34          10


    The trivial_change_matrix
    Sample of trivial_change_matrix:
                        Stock A     Stock B     Stock C
        -> 2018-08-08    34/17       30/10       10/10
        -> 2018-08-09    30/34       34/30       10/10

    The trivial_ranking_matrix = rank(cross-stocks forum matrix)
    Sample of ranking_trivial_matrix:
                        Stock A     Stock B     Stock C
        -> 2018-08-08      2           1            3
        -> 2018-08-09      3           1            2

    The dynamic weighting scheme will be
                weight of A at_time_t = stock A's change at_time_t / sum(all stocks changes, at_time_t)
    Sample of dynamic_weights:
                        Stock A     Stock B     Stock C
        -> 2018-08-08     1/3         1/2          1/6
        -> 2018-08-09     0.29        0.38         0.33

    The equal weighting scheme will be
    Sample of equal_weights:
                        Stock A     Stock B     Stock C
        -> 2018-08-08     1/3         1/3          1/3
        -> 2018-08-09     1/3         1/3          1/3

    """

    def __init__(self, start='2015-01-01', end='2019-07-31', number_of_days_for_averaging=20):
        self._start = dt.datetime.strptime(start, '%Y-%m-%d')
        self._end = dt.datetime.strptime(end, '%Y-%m-%d')
        self.date_list = [self._end - dt.timedelta(idx) for idx in range((self._end - self._start).days + 1)]
        self.date_list.reverse()
        self.number_of_days_for_averaging = number_of_days_for_averaging
        self.preprocess()

    def preprocess(self):
        try:
            self.stocks_post_matrix = pd.read_csv('data/interim/weekend_stocks_post_matrix.csv',
                                                  index_col=0, parse_dates=True)
        except:
            files = os.listdir('data/historical/2019-10-15')
            files = [file for file in files if re.match('[\d]+.csv', file)]
            files = sorted(files, key=lambda x: int(x.split('.')[0]))
            tickers = [file.split('.')[0] for file in files]
            self.stocks_post_matrix = pd.DataFrame(index=self.date_list)
            with Bar('CrossSignal Preprocessing', max=len(files)) as bar:
                for ticker, file in zip(tickers, files):
                    bar.next()
                    df = pd.read_csv(f'data/historical/2019-10-15/{ticker}.csv',
                                     index_col=0, parse_dates=True, low_memory=False)
                    df = df.resample('3H').count()
                    df.reset_index(inplace=True)

                    # We need to set hour column to select the period between 9AM:3PM
                    df['Hour'] = df['Time'].apply(lambda x: x.hour)
                    df.loc[(df['Hour'] < 15) & (df['Hour'] >= 9), 'Title'] = 0

                    # Resample with a base = 15 and freq = 24H
                    # Since we have made 9AM-3PM = 0, so we can sum the 24 hour starting from 3PM
                    curr_stock_posts_vec = pd.Series(df['Title'].values, index=df.Time, name=ticker)
                    curr_stock_posts_vec = curr_stock_posts_vec.groupby(pd.Grouper(freq='24H', base=15)).sum()

                    # This is to make the Hour from 15 to 0AM
                    curr_stock_posts_vec = curr_stock_posts_vec.resample('D').sum()

                    # Critical step is to add one day to the vector since this will be used to
                    # predict the return of next day
                    curr_stock_posts_vec.index = pd.DatetimeIndex(curr_stock_posts_vec.index) + pd.DateOffset(1)

                    # This is to fix the date range
                    curr_stock_posts_vec = curr_stock_posts_vec[
                        (curr_stock_posts_vec.index >= self._start) & (curr_stock_posts_vec.index <= self._end)]
                    self.stocks_post_matrix = pd.concat([self.stocks_post_matrix, curr_stock_posts_vec], axis=1)

            # TODO I change the filled value to 0 from 1, we need to confirm the impact
            self.stocks_post_matrix = self.stocks_post_matrix.fillna(0)
            self.stocks_post_matrix.to_csv('data/interim/stocks_post_matrix.csv')

            """Handle the weekend posts"""
            self.add_weekend_posts()
            self.stocks_post_matrix.to_csv('data/interim/weekend_stocks_post_matrix.csv')

    def add_weekend_posts(self):
        prices_matrix = pd.read_csv('data/interim/prices_matrix.csv', index_col=0, parse_dates=True, header=[0, 1])
        prices_matrix = pd.concat([pd.DataFrame(index=self.date_list), prices_matrix], axis=1)
        holiday_vector = prices_matrix.isnull().all(axis=1)

        is_in_holiday = False
        cum_posts = []
        for idx, row in enumerate(self.stocks_post_matrix.index):
            if is_in_holiday:
                if holiday_vector.loc[row]:
                    cum_posts.append(idx)
                else:
                    for prev in cum_posts:
                        self.stocks_post_matrix.iloc[idx, :] += self.stocks_post_matrix.iloc[prev, :]
                    cum_posts = []
                    is_in_holiday = False
            else:
                if holiday_vector.loc[row]:
                    is_in_holiday = True
                    cum_posts.append(idx)

    @property
    def trivial_change_matrix(self):
        return (self.stocks_post_matrix / self.stocks_post_matrix.shift(1)).replace([np.inf, -np.inf], np.nan)

    @property
    def ranking_trivial_matrix(self):
        return self.trivial_change_matrix.rank(1, method='first', ascending=True)

    @property
    def log_change_matrix(self):
        ave_matrix = self.stocks_post_matrix.rolling(window=self.number_of_days_for_averaging).mean()
        log_matrix = np.log(ave_matrix)
        return (self.trivial_change_matrix * log_matrix).replace([np.inf, -np.inf], np.nan)

    @property
    def log_ranking_matrix(self):
        return self.log_change_matrix.rank(1, method='first', ascending=True)

    @property
    def equal_weight_rank_signal(self):
        daily_post_rank_matrix = self.stocks_post_matrix.rank(1, method='first', ascending=True)
        rank_max = (daily_post_rank_matrix.max(axis=1) * 0.95).round(0)  # The place to change percentage
        daily_post_rank_matrix = daily_post_rank_matrix.gt(rank_max, axis=0)

        daily_post_change_rank_matrix = self.ranking_trivial_matrix
        change_rank_max = (daily_post_change_rank_matrix.max(axis=1) * 0.75).round(0)
        daily_post_change_rank_matrix = daily_post_change_rank_matrix.gt(change_rank_max, axis=0)

        # At here, True: buy False: No position
        whether_to_buy_matrix = daily_post_rank_matrix | daily_post_change_rank_matrix

        weights = whether_to_buy_matrix.apply(lambda row: row / 1, axis=1)  # return 1 means buy and 0 to sell
        return weights


class Backtest:
    """
    Backtest the trading signal for a single stock. We assume that the number of posts at day t minus the number of
    posts at day t - 1 will contribute to the return at day t.
    """

    def __init__(self, signal: pd.DataFrame, start='2015-01-01', end='2019-07-31', number_of_skipping_days=60,
                 ret_type='cmc_ret'):
        self._signal = signal.fillna(0)
        self._tickers = signal.columns.values.tolist()
        self._start = dt.datetime.strptime(start, '%Y-%m-%d')
        self._end = dt.datetime.strptime(end, '%Y-%m-%d')
        self.date_list = [self._end - dt.timedelta(idx) for idx in range((self._end - self._start).days + 1)]
        self.number_of_skipping_days = number_of_skipping_days  # To skip the IPO period
        self.preprocess(ret_type)
        self.cash = 10_000_000
        self.equity_value = 0
        self.total_value = 0
        self.cash_history = []
        self.equity_value_history = []
        self.total_value_history = []
        self.individual_history = []
        self.valid_dates = []
        self.lot_size = 100
        self._ret_type = ret_type

    @staticmethod
    def number_of_available_data_in_row(df):
        df['available_data_in_row'] = df.count(axis=1)
        return df

    def preprocess(self, ret_type):
        """Build a multi-indexed matrix with ticker as level 1, [close, open, ret] as level 2, timestamp as level 3"""
        for ticker in self._tickers:
            ticker = str(ticker)
            curr_df = pd.read_csv(f'data/prices/{ticker}.csv', index_col=0, low_memory=False,
                                  names=pd.MultiIndex.from_product([[ticker], ['open', 'close']]))[1:]
            curr_df.index = pd.to_datetime(curr_df.index)
            curr_df = curr_df[(curr_df.index >= self._start) & (curr_df.index <= self._end)]
            curr_df = curr_df.astype(float)
            curr_df.loc[:, (ticker, 'cmo_ret')] = (curr_df.loc[:, (ticker, 'close')] -
                                                   curr_df.loc[:, (ticker, 'open')]) / \
                                                  curr_df.loc[:, (ticker, 'open')]
            curr_df.loc[:, (ticker, 'cmc_ret')] = (curr_df.loc[:, (ticker, 'close')] -
                                                   curr_df.loc[:, (ticker, 'close')].shift(1)) / \
                                                  curr_df.loc[:, (ticker, 'close')].shift(1)
            curr_df.loc[:, (ticker, 'omo_ret')] = (curr_df.loc[:, (ticker, 'open')] -
                                                   curr_df.loc[:, (ticker, 'open')].shift(1)) / \
                                                  curr_df.loc[:, (ticker, 'open')].shift(1)
            curr_df.loc[:, (ticker, 'abs_diff_oc')] = np.abs((curr_df.loc[:, (ticker, 'close')] -
                                                  curr_df.loc[:, (ticker, 'open')]))
            try:
                self.prices_matrix = pd.concat([self.prices_matrix, curr_df], axis=1)
            except:
                self.prices_matrix = curr_df

        # self.prices_matrix.fillna(0, inplace=True)
        self.prices_matrix.to_csv('data/interim/prices_matrix.csv')
        # Choose the preferred ret
        self.ret_matrix = self.prices_matrix.iloc[:, self.prices_matrix.columns.get_level_values(1) == ret_type]

        # Critical step to drop level of columns
        self.ret_matrix.columns = self.ret_matrix.columns.get_level_values(0)

        # Match the dates list
        self.ret_matrix = pd.concat([pd.DataFrame(index=self.date_list), self.ret_matrix], axis=1)
        self.yesterday_inventory = pd.Series(np.repeat(0, self.ret_matrix.shape[1]),
                                             index=self.ret_matrix.columns)

    @property
    def tradability_matrix(self):
        """
        This function will do three things
            1.  Skip IPO periods $DONE$
            2.  Skip the day if the stock increase/decrease by 10%  $TODO$
            3.  Skip holiday, halted dates $DONE$
        :return: tradability_matrix in which values are either 1 or 0
        """
        ret_matrix = deepcopy(self.ret_matrix)

        """Here will exclude the IPO prices by replacing the first n available prices into Zero"""
        for col in ret_matrix.columns:
            valid_index = ret_matrix[col].first_valid_index()
            ret_matrix.loc[[valid_index + dt.timedelta(idx)
                            for idx in range(self.number_of_skipping_days)], col] = np.nan

        """Here will deal with the price limit case"""
        prices_matrix = deepcopy(self.prices_matrix)
        # Here I used (close - open) / open as the indicator, later we may change to ohlcv with new dataset
        prices_diff_mat = prices_matrix.iloc[:, prices_matrix.columns.get_level_values(1) == 'cmo_ret']
        prices_diff_mat.columns = prices_diff_mat.columns.get_level_values(0)
        prices_diff_mat = prices_diff_mat[ret_matrix.columns]  # Critical step to make sure data in the same order
        is_prices_limit_mat = pd.DataFrame(np.where(np.abs(prices_diff_mat.values) < 0.002, np.nan, 1),
                                           index=prices_diff_mat.index, columns=prices_diff_mat.columns)
        is_prices_limit_mat = pd.concat([pd.DataFrame(index=self.date_list), is_prices_limit_mat], axis=1)
        after_price_limit_ret_matrix = ret_matrix * is_prices_limit_mat

        return pd.DataFrame(np.where(np.isnan(after_price_limit_ret_matrix), 0, 1),
                            index=ret_matrix.index,
                            columns=ret_matrix.columns).fillna(0)

    def simulate(self):
        """
        Question about backtesting
            1.  How we trade? For now, I will buy a fixed amount of value
            2.
        """
        self.ret_matrix.round(3).to_csv('data/interim/ret_matrix.csv')
        self.tradability_matrix.to_csv('data/interim/tradability.csv')
        self._signal.to_csv('data/interim/signal_matrix.csv')
        with Bar('Backtesting', max=self.ret_matrix.shape[0]) as bar:
            for idx, date in enumerate(self.ret_matrix.index):
                bar.next()
                today_signals = self._signal.loc[date, :]
                today_rets = self.ret_matrix.loc[date, :]
                today_rets = today_rets.fillna(0)
                today_tradability = self.tradability_matrix.loc[date, :]

                """
                Backtesting
                
                Suppose we are at day N
                Assumption:
                    1. We can cash out the holdings by using the prices of day N - 1 at day N.
                    2. We can only long the equities
                    3. We can buy the stocks at the close price of day N
                """

                # Any none zero will equal to one
                # 1 or 0 -> 1,
                # 0 or 1 -> 1,
                # 1 or 1 -> 1,
                # 0 or 0 -> 0
                # today_position is different from the updated inventory position since we assume
                # we will get profit from yesturday + today_signals * today_tradability today
                # but the remaining invection is updated by self.update_inventory(today_signals, today_tradability)
                # today_position = pd.DataFrame([self.yesterday_inventory.values,
                #                                 (today_signals * today_tradability).values]).any(axis=0)

                """only inventory will be used to compute return"""
                today_position = self.yesterday_inventory
                self.update_inventory(today_signals, today_tradability)  # Update inventory

                if not (np.all(today_position == 0) or np.all(today_rets == 0)):
                    self.total_value += np.sum(today_rets.values * today_position.values)
                    self.total_value_history.append(self.total_value)
                    self.valid_dates.append(date)
                    self.individual_history.append(today_rets.values * today_position.values)
        self.pye_plot()

    def update_inventory(self, today_signals, today_tradability):
        """Only update the inventory when the tradability is True"""
        for idx in range(self.yesterday_inventory.size):
            if today_tradability[idx]:
                self.yesterday_inventory[idx] = today_signals[idx]

    def plot(self):

        plot_data = pd.DataFrame(self.individual_history, columns=self.ret_matrix.columns,
                                 index=self.valid_dates)
        plot_data = pd.concat([plot_data, pd.Series(self.total_value_history, name='Total',
                                                    index=self.valid_dates)], axis=1)
        plot_data.to_csv(f'data/interim/individual_pnl_{self._ret_type}.csv')
        plot_data.plot()
        plt.legend()
        # plt.plot(self.valid_dates, self.total_value_history, color='red', linewidth=2, label="PnL")
        plt.show()

    def pye_plot(self):
        plot_data = pd.DataFrame(self.individual_history, columns=self.ret_matrix.columns,
                                 index=self.valid_dates)
        plot_data = pd.concat([plot_data, pd.Series(self.total_value_history, name='Total',
                                                    index=self.valid_dates)], axis=1)
        plot_data = plot_data.round(3)
        plot_data.to_csv(f'data/interim/individual_pnl_{self._ret_type}.csv')

        line = Line(init_opts=opts.InitOpts(width="1200px", height="600px"))
        line.add_xaxis(
            xaxis_data=[date.strftime('%Y-%m-%d') for date in
                        self.valid_dates])  # input of x-axis has been string format
        for col in plot_data.columns:
            if col == 'Total':
                line.add_yaxis(y_axis=plot_data.loc[:, col].values.tolist(),
                               series_name=col.title(),
                               is_smooth=True,
                               label_opts=opts.LabelOpts(is_show=False),
                               linestyle_opts=opts.LineStyleOpts(width=2)
                               )
        line.set_global_opts(
            datazoom_opts=opts.DataZoomOpts(),
            legend_opts=opts.LegendOpts(pos_top="20%", pos_right='0%', pos_left='90%'),
            title_opts=opts.TitleOpts(title='Total Returns'.upper(), pos_left='0%'),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross", is_show=True),
            xaxis_opts=opts.AxisOpts(boundary_gap=False, max_interval=5),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            )
        )
        # line.set_series_opts(
        #     markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='max', name='Max'),
        #                                             opts.MarkPointItem(type_='min', name='Min')]),
        # )
        line.render(path=f'figs/Total Returns ({self._ret_type}).html')


def sentiment(texts: str):
    text = SnowNLP(texts)
    sents = text.sentences
    for sen in sents:
        s = SnowNLP(sen)
        print(sen, '-', s.sentiments)


def run_backtest():
    start = '2015-01-01'
    end = '2019-07-31'
    cs = CrossSignal(start=start, end=end)
    # print(cs.equal_weight_rank_signal)

    bs = Backtest(cs.equal_weight_rank_signal, start=start, end=end, ret_type='cmo_ret')
    bs.simulate()


if __name__ == '__main__':
    run_backtest()

    # df = pd.read_csv(f'/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/'
    #                  f'Project_2-Forum/data/historical/2019-10-15/{600010}.csv')
    # real_text = df.loc[:20, 'Title'].values.tolist()
    # text = '快买这个股票，这个股票一定能够大涨，这个不是太好用，' \
    #        '超级垃圾股票，绝对的优质股票，买这股票就是去送钱，一定亏钱的，这个股票不太好可能亏钱'
    # sentiment(texts=text)
    # sentiment('，'.join(real_text))
