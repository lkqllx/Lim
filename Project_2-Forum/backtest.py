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
import sys
import jqdatasdk as jq
import logging


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

    def __init__(self, start='2015-01-01', end='2019-07-31', number_of_days_for_averaging=20, signal_period=1,
                 decile=1, sentiment=None):
        self._start = dt.datetime.strptime(start, '%Y-%m-%d')
        self._end = dt.datetime.strptime(end, '%Y-%m-%d')
        self.date_list = [self._end - dt.timedelta(idx) for idx in range((self._end - self._start).days + 1)]
        self.date_list.reverse()
        self.number_of_days_for_averaging = number_of_days_for_averaging
        self.signal_period = signal_period
        self.decile = decile
        self.sentiment = sentiment
        self.preprocess()

    def preprocess(self):
        try:
            if self.sentiment is None:
                self.stocks_post_matrix = pd.read_csv('data/interim/weekend_stocks_post_matrix_after_masking.csv',
                                                      index_col=0, parse_dates=True)
            elif self.sentiment == 'positive':
                self.stocks_post_matrix = pd.read_csv('data/interim/weekend_stocks_post_matrix_after_masking_positive.csv',
                                                      index_col=0, parse_dates=True)
            else:
                self.stocks_post_matrix = pd.read_csv('data/interim/weekend_stocks_post_matrix_after_masking_negative.csv',
                                                      index_col=0, parse_dates=True)

        except:
            files = os.listdir('data/historical/sentiment')
            files = [file for file in files if re.match('[\d]+.csv', file)]
            files = sorted(files, key=lambda x: int(x.split('.')[0]))
            tickers = [file.split('.')[0] for file in files]
            self.stocks_post_matrix = pd.DataFrame(index=self.date_list)
            with Bar('CrossSignal Preprocessing', max=len(files)) as bar:
                for ticker, file in zip(tickers, files):
                    bar.next()
                    df = pd.read_csv(f'data/historical/sentiment/{ticker}.csv',
                                     index_col=0, parse_dates=True, low_memory=False)
                    if self.sentiment is None:
                        pass
                    elif self.sentiment == 'positive':
                        df = df[df['sent_label'] == 1]
                    else:
                        df = df[df['sent_label'] == 0]
                    last_date = df.index[-1]
                    # df = df.resample('3H').count()
                    # df = df[df.index >= last_date]
                    # df.reset_index(inplace=True)
                    #
                    # # We need to set hour column to select the period between 9AM:3PM
                    # df['Hour'] = df['Time'].apply(lambda x: x.hour)
                    # df.loc[(df['Hour'] < 15) & (df['Hour'] >= 9), 'Title'] = 0

                    # Resample with a base = 15 and freq = 24H
                    # Since we have made 9AM-3PM = 0, so we can sum the 24 hour starting from 3PM
                    # curr_stock_posts_vec = pd.Series(df['Title'].values, index=df.Time, name=ticker)
                    curr_stock_posts_vec = df.groupby(pd.Grouper(freq='60T', base=30)).count()['Title']
                    curr_stock_posts_vec.name = ticker
                    curr_stock_posts_vec = curr_stock_posts_vec.groupby(pd.Grouper(freq='24H', base=14)).sum()

                    # This is to make the Hour from 15 to 0AM
                    curr_stock_posts_vec = curr_stock_posts_vec.resample('D').sum()

                    # Critical step is to add one day to the vector since this will be used to
                    # predict the return of next day
                    curr_stock_posts_vec.index = pd.DatetimeIndex(curr_stock_posts_vec.index) + pd.DateOffset(1)

                    # This is to fix the date range
                    curr_stock_posts_vec = curr_stock_posts_vec[
                        (curr_stock_posts_vec.index >= self._start) & (curr_stock_posts_vec.index <= self._end)]
                    self.stocks_post_matrix = pd.concat([self.stocks_post_matrix, curr_stock_posts_vec], axis=1)

                    # self.stocks_post_matrix = self.stocks_post_matrix.fillna(0)
                # self.create_csi300_mask_matrix()
                if self.sentiment is None:
                    self.stocks_post_matrix.to_csv('data/interim/stocks_post_matrix.csv')

                    """Handle the weekend posts"""
                    self.add_weekend_posts()
                    self.delete_zeros_weekend()
                    self.stocks_post_matrix.to_csv('data/interim/weekend_stocks_post_matrix.csv')
                    self.create_csi300_mask_matrix()
                    self.stocks_post_matrix.to_csv('data/interim/weekend_stocks_post_matrix_after_masking.csv')
                else:
                    self.stocks_post_matrix.to_csv(f'data/interim/stocks_post_matrix_{self.sentiment.lower()}.csv')

                    """Handle the weekend posts"""
                    self.add_weekend_posts()
                    self.delete_zeros_weekend()
                    self.stocks_post_matrix.to_csv(f'data/interim/weekend_stocks_post_matrix_{self.sentiment.lower()}.csv')
                    self.create_csi300_mask_matrix()
                    self.stocks_post_matrix.to_csv(f'data/interim/weekend_stocks_post_matrix_'
                                                   f'after_masking_{self.sentiment.lower()}.csv')

    def delete_zeros_weekend(self):
        for col in self.stocks_post_matrix.columns:
            curr_series = self.stocks_post_matrix.loc[:, col].dropna()
            curr_series = curr_series[curr_series != 0]
            first_valid_index = curr_series.index[0]
            self.stocks_post_matrix.loc[:, col][self.stocks_post_matrix.index < first_valid_index] = np.nan


    def add_weekend_posts(self):
        """Deal with weekend cases that the posts should be aggregated into next trading day"""
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
                        self.stocks_post_matrix.iloc[prev, :] = 0
                    cum_posts = []
                    is_in_holiday = False
            else:
                if holiday_vector.loc[row]:
                    is_in_holiday = True
                    cum_posts.append(idx)

    def trivial_change_matrix(self):
        stocks_post_matrix = pd.read_csv('data/interim/stocks_post_matrix.csv',  index_col=0, parse_dates=True)
        change_matrix = deepcopy(stocks_post_matrix.dropna(how='all'))
        change_matrix = change_matrix.rolling(f'{self.signal_period}D',min_periods=self.signal_period).sum()
        change_rank_matrix = change_matrix.rank(1, method='first', ascending=True)
        # change_matrix = (change_matrix / change_matrix.shift(1)).replace([np.inf, -np.inf], np.nan)
        return (change_rank_matrix - change_rank_matrix.shift(1)).replace([np.inf, -np.inf], np.nan)

    def ranking_trivial_matrix(self):
        return self.trivial_change_matrix().rank(1, method='first', ascending=True)

    @property
    def log_change_matrix(self):
        ave_matrix = self.stocks_post_matrix.rolling(window=self.number_of_days_for_averaging).mean()
        log_matrix = np.log(ave_matrix)
        return (self.trivial_change_matrix * log_matrix).replace([np.inf, -np.inf], np.nan)

    @property
    def log_ranking_matrix(self):
        return self.log_change_matrix.rank(1, method='first', ascending=True)

    def create_csi300_mask_matrix(self):
        """This function is designed to process the 490 * 490 post matrix to a format that
        each row only contains 300 constituents at that time"""
        if not os.path.exists('data/interim/masking_matrix.csv'):
            effective_dates = ['31-07-2019','17-06-2019', '17-12-2018', '11-06-2018', '11-12-2017', '12-06-2017',
                               '12-12-2016', '13-06-2016', '30-12-2015', '30-11-2015', '15-06-2015', '14-05-2015',
                               '26-01-2015', '01-01-2015']
            effective_dates = list(map(dt.datetime.strptime, effective_dates, ['%d-%m-%Y'] * len(effective_dates)))
            effective_dates.reverse()

            """This mask matrix will be used to multiply with stocks_post_matrix"""
            mask_matrix = pd.DataFrame(np.nan, index=self.date_list, columns=self.stocks_post_matrix.columns)

            jq.auth('18810906018', '906018')
            with Bar('Masking', max=len(effective_dates)) as bar:
                for idx, date in enumerate(effective_dates):
                    bar.next()
                    curr_stocks = jq.get_index_stocks('000300.XSHG', date)
                    curr_stocks = [stock.split('.')[0] for stock in curr_stocks]
                    if idx != len(effective_dates) - 1:
                        """if not last date, we will assign 1 to available stocks until the Previous 
                        day of next next effective day"""
                        mask_matrix.loc[date:effective_dates[idx+1]-dt.timedelta(1), curr_stocks] = 1
                    else:
                        mask_matrix.loc[effective_dates[idx], curr_stocks] = 1
            mask_matrix.to_csv('data/interim/masking_matrix.csv')
        else:
            mask_matrix = pd.read_csv('data/interim/masking_matrix.csv', index_col=0, parse_dates=True)
        self.stocks_post_matrix = self.stocks_post_matrix.fillna(0)
        self.stocks_post_matrix = mask_matrix * self.stocks_post_matrix


    def equal_weight_rank_signal(self):
        stocks_post_matrix = self.stocks_post_matrix.rolling(f'{self.signal_period}D',
                                                                  min_periods=self.signal_period).sum()
        curr_post_matrix = deepcopy(stocks_post_matrix)
        daily_post_rank_matrix = curr_post_matrix.rank(1, method='first', ascending=True)

        rank_max = (daily_post_rank_matrix.max(axis=1) * self.decile * 0.1).round(0)  # The place to change percentage
        rank_min = (daily_post_rank_matrix.max(axis=1) * (self.decile - 1) * 0.1).round(0)
        daily_post_rank_matrix = daily_post_rank_matrix.gt(rank_min, axis=0) & \
                                 daily_post_rank_matrix.le(rank_max, axis=0)

        daily_post_change_rank_matrix = deepcopy(self.ranking_trivial_matrix())
        change_rank_max = (daily_post_change_rank_matrix.max(axis=1) * self.decile * 0.1).round(0)
        change_rank_min = (daily_post_change_rank_matrix.max(axis=1) * (self.decile - 1) * 0.1).round(0)
        daily_post_change_rank_matrix = daily_post_change_rank_matrix.gt(change_rank_min, axis=0) & \
                                        daily_post_change_rank_matrix.le(change_rank_max, axis=0)

        # constraints = self.constraints('CAP')
        # constraints = pd.concat([pd.DataFrame(index=daily_post_rank_matrix.index), constraints], axis=1)
        # constraints.fillna(False, inplace=True)
        # At here, True: buy False: No position
        # whether_to_buy_matrix = daily_post_rank_matrix | daily_post_change_rank_matrix
        whether_to_buy_matrix = daily_post_rank_matrix

        weights = whether_to_buy_matrix.apply(lambda row: row / 1, axis=1)  # return 1 means buy and 0 to sell
        return weights

    def equal_weight_decile_long_short(self):
        stocks_post_matrix = self.stocks_post_matrix
        curr_post_matrix = deepcopy(stocks_post_matrix)
        daily_post_rank_matrix = curr_post_matrix.rank(1, method='first', ascending=True)

        rank_max = (daily_post_rank_matrix.max(axis=1) * 1 * 0.1).round(0)  # The place to change percentage
        rank_min = (daily_post_rank_matrix.max(axis=1) * (1 - 1) * 0.1).round(0)
        long_daily_post_rank_matrix = daily_post_rank_matrix.gt(rank_min, axis=0) & \
                                 daily_post_rank_matrix.le(rank_max, axis=0)
        long_weights = long_daily_post_rank_matrix.apply(lambda row: row / 1, axis=1)

        rank_max = (daily_post_rank_matrix.max(axis=1) * 10 * 0.1).round(0)  # The place to change percentage
        rank_min = (daily_post_rank_matrix.max(axis=1) * (10 - 1) * 0.1).round(0)
        short_daily_post_rank_matrix = daily_post_rank_matrix.gt(rank_min, axis=0) & \
                                 daily_post_rank_matrix.le(rank_max, axis=0)

        short_weights = short_daily_post_rank_matrix.apply(lambda row: row / 1, axis=1)
        weights = long_weights - short_weights
        return weights

    def constraints(self, constraint_type):
        if constraint_type == 'MA':
            prices_matrix = pd.read_csv('data/interim/prices_matrix.csv', index_col=0, parse_dates=True, header=[0, 1])
            prices_matrix = prices_matrix.loc[:, prices_matrix.columns.get_level_values(1) == 'close']
            prices_matrix.columns = prices_matrix.columns.get_level_values(0)
            rolling_mean = prices_matrix.rolling(window=20, min_periods=1, axis=0).mean()
            difference_matrix = (prices_matrix - rolling_mean).shift(1).fillna(False)
            stocks_std = difference_matrix.std()
            signal_matrix = difference_matrix.apply(lambda row: row >= stocks_std, axis=1)
        elif constraint_type == 'CAP':
            cap_matrix = pd.read_csv('data/fundamental/market_caps.csv', index_col=0, parse_dates=True)
            cap_matrix = cap_matrix[(cap_matrix.index >= self._start) & (cap_matrix.index <= self._end)]
            quantile = cap_matrix.quantile(0.33, axis=1)
            signal_matrix = cap_matrix.ge(quantile, axis=0)
        return signal_matrix


    @property
    def low_rank_equal_weight_signal(self):
        daily_post_rank_matrix = self.stocks_post_matrix.rank(1, method='first', ascending=True)
        rolling_daily_post_rank_matrix = daily_post_rank_matrix.rolling(window=20, axis=0).mean()
        rank_max = (daily_post_rank_matrix.max(axis=1) * 0.3).round(0)  # The place to change percentage
        daily_post_rank_matrix = rolling_daily_post_rank_matrix.lt(rank_max, axis=0)

        daily_post_change_rank_matrix = self.ranking_trivial_matrix
        change_rank_max = (daily_post_change_rank_matrix.max(axis=1) * 0.9).round(0)
        daily_post_change_rank_matrix = daily_post_change_rank_matrix.lt(change_rank_max, axis=0)

        # At here, True: buy False: No position
        whether_to_buy_matrix = daily_post_rank_matrix & daily_post_change_rank_matrix

        weights = whether_to_buy_matrix.apply(lambda row: row / 1, axis=1)  # return 1 means buy and 0 to sell
        return weights


class Backtest:
    """
    Backtest the trading signal for a single stock. We assume that the number of posts at day t minus the number of
    posts at day t - 1 will contribute to the return at day t.
    """

    def __init__(self, signal: pd.DataFrame, start='2015-01-01', end='2019-07-31', number_of_skipping_days=60,
                 number_of_prices_delay=2, path=None):
        self._signal = signal
        self._tickers = signal.columns.values.tolist()
        self._start = dt.datetime.strptime(start, '%Y-%m-%d')
        self._end = dt.datetime.strptime(end, '%Y-%m-%d')
        self.date_list = [self._end - dt.timedelta(idx) for idx in range((self._end - self._start).days + 1)]
        self.number_of_skipping_days = number_of_skipping_days  # To skip the IPO period
        self.cash = 10_000_000
        self.equity_value = 0
        self.total_value = 1
        self.total_value_tranc = 1
        self.cash_history = []
        self.equity_value_history = []
        self.total_value_history = []
        self.total_value_tranc_history = []
        self.individual_history = []
        self.inventory_history = []
        self.valid_dates = []
        self.inventory_dates = []
        self.different_start_daily_returns = []
        self.different_start_daily_returns_transac = []
        self.lot_size = 100
        self.number_of_prices_delay = number_of_prices_delay
        self.path = path
        self.preprocess()

    @staticmethod
    def number_of_available_data_in_row(df):
        df['available_data_in_row'] = df.count(axis=1)
        return df

    def preprocess(self):
        """Build a multi-indexed matrix with ticker as level 1, [close, open, ret] as level 2, timestamp as level 3"""

        csi300 = pd.read_csv('data/target_list/csi300_prices.csv', index_col=0, parse_dates=True)
        csi300_close = csi300['Price']
        csi300_close = csi300_close.apply(lambda row: float(''.join(re.findall('(\d)+,([\d.]+)', row)[0])))
        self.csi300_close_ret = (csi300_close.shift(1) - csi300_close) / csi300_close
        self.csi300_close_ret.name = f'CSI300_cmc1'
        try:
            self.prices_matrix = pd.read_csv('data/interim/prices_matrix.csv', index_col=0, parse_dates=True,
                                             header=[0, 1])
        except:
            with Bar('Concatenating Returns', max=len(self._tickers)) as bar:
                for ticker in self._tickers:
                    bar.next()
                    ticker = str(ticker)
                    curr_df = pd.read_csv(f'data/prices/{ticker}.csv', index_col=0, low_memory=False,
                                          names=pd.MultiIndex.from_product([[ticker], ['open', 'close']]))[1:]
                    curr_df.index = pd.to_datetime(curr_df.index)
                    curr_df = curr_df[(curr_df.index >= self._start) & (curr_df.index <= self._end)]
                    curr_df = curr_df.astype(float)
                    curr_df.loc[:, (ticker, 'cmo_ret')] = (curr_df.loc[:, (ticker, 'close')] -
                                                           curr_df.loc[:, (ticker, 'open')]) / \
                                                          curr_df.loc[:, (ticker, 'open')]
                    curr_df.loc[:, (ticker, 'omo_ret')] = (curr_df.loc[:, (ticker, 'open')] -
                                                           curr_df.loc[:, (ticker, 'open')].shift(1)) / \
                                                          curr_df.loc[:, (ticker, 'open')].shift(1)
                    for idx in range(1, self.number_of_prices_delay):
                        curr_df.loc[:, (ticker, f'cmc{idx}_ret')] = (curr_df.loc[:, (ticker, 'close')] -
                                                                     curr_df.loc[:, (ticker, 'close')].shift(idx)) / \
                                                                    curr_df.loc[:, (ticker, 'close')].shift(idx)

                    curr_df.loc[:, (ticker, 'ompc_ret')] = (curr_df.loc[:, (ticker, 'open')] -
                                                                 curr_df.loc[:, (ticker, 'close')].shift(-1)) / \
                                                                curr_df.loc[:, (ticker, 'close')].shift(-1)

                    try:
                        self.prices_matrix = pd.concat([self.prices_matrix, curr_df], axis=1)
                    except:
                        self.prices_matrix = curr_df

                # self.prices_matrix.fillna(0, inplace=True)
        # self.prices_matrix.to_csv('data/interim/prices_matrix.csv')

        # Choose the preferred ret
        self.ret_matrix = self.prices_matrix.iloc[:, self.prices_matrix.columns.get_level_values(1) == 'cmc1_ret']
        self.close_matrix = self.prices_matrix.iloc[:, self.prices_matrix.columns.get_level_values(1) == 'close']

        # Critical step to drop level of columns
        self.ret_matrix.columns = self.ret_matrix.columns.get_level_values(0)
        self.close_matrix.columns = self.close_matrix.columns.get_level_values(0)

        # Match the dates list
        self.ret_matrix = pd.concat([pd.DataFrame(index=self.date_list), self.ret_matrix], axis=1)
        self.close_matrix = pd.concat([pd.DataFrame(index=self.date_list), self.close_matrix], axis=1)


    @property
    def tradability_matrix(self):
        """
        This function will do three things
            1.  Skip IPO periods $DONE$
            2.  Skip the day if the stock increase/decrease by 10%
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
        intraday_diff_mat = prices_matrix.iloc[:, prices_matrix.columns.get_level_values(1) == 'cmo_ret']
        intraday_diff_mat.columns = intraday_diff_mat.columns.get_level_values(0)
        intraday_diff_mat = intraday_diff_mat[ret_matrix.columns]  # Critical step to make sure data in the same order

        interday_diff_mat = prices_matrix.iloc[:, prices_matrix.columns.get_level_values(1) == 'ompc_ret']
        interday_diff_mat.columns = interday_diff_mat.columns.get_level_values(0)
        interday_diff_mat = interday_diff_mat[ret_matrix.columns]  # Critical step to make sure data in the same order
        is_prices_limit_array = np.where(np.logical_and(np.abs(intraday_diff_mat) < 0.002,
                                                        np.abs(interday_diff_mat) > 0.095), np.nan, 1)
        is_prices_limit_mat = pd.DataFrame(is_prices_limit_array,
                                           index=intraday_diff_mat.index,
                                           columns=intraday_diff_mat.columns)
        is_prices_limit_mat = pd.concat([pd.DataFrame(index=self.date_list), is_prices_limit_mat], axis=1)
        after_price_limit_ret_matrix = ret_matrix * is_prices_limit_mat

        return pd.DataFrame(np.where(np.isnan(after_price_limit_ret_matrix), 0, 1),
                            index=ret_matrix.index,
                            columns=ret_matrix.columns).fillna(0)

    def reset(self):
        self.total_value = 1
        self.total_value_tranc = 1
        self.cash_history = []
        self.equity_value_history = []
        self.total_value_history = []
        self.total_value_tranc_history = []
        self.individual_history = []
        self.inventory_history = []
        self.valid_dates = []
        self.inventory_dates = []
        self.yesterday_inventory = pd.Series(np.repeat(0, self.ret_matrix.shape[1]),
                                             index=self.ret_matrix.columns)

    def init_backtest(self, interval):
        self.reset()
        self._ret_type = f'cmc{interval}_ret'

        # self.ret_matrix.round(3).to_csv('data/interim/ret_matrix.csv')
        # self._signal.to_csv('data/interim/signal_matrix.csv')

    def simulate_one_portfolio(self, start_date=0, interval=10):
        """Becktesting function for holding one portfolio at a time"""
        self.init_backtest(interval=interval)
        tradability_mat = self.tradability_matrix
        # tradability_mat.to_csv('data/interim/tradability.csv')
        first_time_flag = True
        with Bar(f'Backtesting {self._ret_type.upper()} - {start_date}', max=self.ret_matrix.shape[0]) as bar:
            for idx, date in enumerate(self.ret_matrix.index):
                bar.next()
                today_signals = self._signal.loc[date, :]
                today_rets = self.ret_matrix.loc[date, :]
                today_rets = today_rets.fillna(0)
                today_tradability = tradability_mat.loc[date, :]


                if np.all(today_tradability == 0):
                    continue

                """only inventory will be used to compute return"""
                today_position = deepcopy(self.yesterday_inventory)

                """Handle the start position"""
                if first_time_flag:
                    trading_date_count = 0
                    first_time_flag = False
                else:
                    trading_date_count += 1

                cost = 0
                today_benchmark = self.csi300_close_ret.loc[date]
                # Update inventory
                if not first_time_flag:
                    if (trading_date_count - start_date) % interval == 0:
                        self.update_inventory(today_signals, today_tradability)
                        cost = self.cal_daily_transaction_cost(np.abs(self.yesterday_inventory), np.abs(today_position))

                total_ret = (today_rets * today_position)
                total_ret = total_ret.dropna()
                if total_ret.sum() != 0:
                    averaged_ret = total_ret[total_ret != 0].mean()
                else:
                    averaged_ret = 0
                today_true_Return = averaged_ret if np.sum(today_position) != 0 else averaged_ret
                self.total_value_history.append(today_true_Return)
                # self.total_value_tranc += averaged_ret - cost
                self.total_value_tranc_history.append(today_true_Return - cost)
                self.valid_dates.append(date)
                self.individual_history.append(today_rets.values * today_position.values)
                self.inventory_history.append(today_position.values)

            self.different_start_daily_returns.append(pd.Series(np.array(self.total_value_history).T,
                                                                name=f'offset {start_date}',
                                                                index=self.valid_dates))

            self.different_start_daily_returns_transac.append(pd.Series(np.array(self.total_value_tranc_history).T,
                                                                        name=f'offset {start_date}',
                                                                        index=self.valid_dates))
            pd.DataFrame(np.array([self.total_value_history, self.total_value_tranc_history]).T, index=self.valid_dates,
                         columns=['averaged daily return', 'averaged daily return after transaction cost']).\
                to_csv(f'{self.path}/{self._ret_type}.csv')
            self.save(save_to=f'{self.path}/individual_pnl_{self._ret_type}_{start_date}.csv')

    @staticmethod
    def cal_daily_transaction_cost(next_pos, today_pos):
        sum_pos = np.sum(today_pos)
        next_sum_pos = np.sum(next_pos)
        if sum_pos != 0:
            turnover = np.sum(np.abs(next_pos - today_pos))/np.sum(today_pos)
        elif (next_sum_pos == 0) and (sum_pos == 0):
            turnover = 0
        else:
            turnover = 1
        return 0.001 * turnover

    def simulate(self, ret_type):

        self.reset()
        # Choose the preferred ret
        self.ret_matrix = self.prices_matrix.iloc[:, self.prices_matrix.columns.get_level_values(1) == ret_type]

        # Critical step to drop level of columns
        self.ret_matrix.columns = self.ret_matrix.columns.get_level_values(0)

        # Match the dates list
        self.ret_matrix = pd.concat([pd.DataFrame(index=self.date_list), self.ret_matrix], axis=1)
        self.yesterday_inventory = pd.Series(np.repeat(0, self.ret_matrix.shape[1]),
                                             index=self.ret_matrix.columns)
        self._ret_type = ret_type
        # self.ret_matrix.round(3).to_csv('data/interim/ret_matrix.csv')
        # self.tradability_matrix.to_csv('data/interim/tradability.csv')
        # self._signal.to_csv('data/interim/signal_matrix.csv')
        tradability_mat = self.tradability_matrix
        with Bar(f'Backtesting {self._ret_type.upper()}', max=self.ret_matrix.shape[0]) as bar:
            for idx, date in enumerate(self.ret_matrix.index):
                bar.next()
                today_signals = self._signal.loc[date, :]
                today_rets = self.ret_matrix.loc[date, :]
                today_rets = today_rets.fillna(0)
                today_tradability = tradability_mat.loc[date, :]

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
                today_position = deepcopy(self.yesterday_inventory)
                # Update inventory
                self.update_inventory(today_signals, today_tradability)

                if self._ret_type == 'cmo_ret':
                    today_position = deepcopy(self.yesterday_inventory)

                if not (np.all(today_position == 0) or np.all(today_rets == 0)):
                    self.total_value += np.sum(today_rets.values * today_position.values)
                    self.total_value_history.append(self.total_value)
                    self.valid_dates.append(date)
                    self.individual_history.append(today_rets.values * today_position.values)
                    self.inventory_history.append(today_position.values)
        self.save(save_to=f'data/interim/ndividual/individual_pnl_{self._ret_type}.csv')

    def update_inventory(self, today_signals, today_tradability):
        """Only update the inventory when the tradability is True"""
        # for idx in range(self.yesterday_inventory.size):
        #     if today_tradability[idx]:
        #         self.yesterday_inventory[idx] = today_signals[idx]
        updated_signal = np.logical_and(today_signals, today_tradability) * today_signals
        unchanged_signal = np.logical_and(self.yesterday_inventory, (1 - today_tradability)) * self.yesterday_inventory
        self.yesterday_inventory = updated_signal + unchanged_signal

    def save(self, save_to):
        plot_data = pd.DataFrame(self.individual_history, columns=self.ret_matrix.columns,
                                 index=self.valid_dates)
        plot_data = pd.concat([plot_data, pd.Series(self.total_value_history, name='Total',
                                                    index=self.valid_dates)], axis=1)
        plot_data = plot_data.round(3)
        plot_data.to_csv(save_to)

        pd.DataFrame(self.inventory_history, index=self.valid_dates,
                     columns=self.ret_matrix.columns).to_csv('data/interim/inventory_history.csv')

    def cal_turnover(self):
        inventory_history = pd.DataFrame(self.inventory_history, index=self.valid_dates,
                                         columns=self.ret_matrix.columns)
        original_row_count = np.count_nonzero(inventory_history, axis=1)
        delay = re.findall('\w+([\d]+).+', self._ret_type)[0]
        diff_mat = inventory_history.diff(periods=int(delay))
        diff_row_count = np.count_nonzero(diff_mat, axis=1)
        turnover_vec = diff_row_count[1:] / original_row_count[1:]
        ave_turnover = np.round(np.mean(turnover_vec), 2)
        print(f'{self._ret_type}s turnover is {ave_turnover}')


def run_backtest():

    """
    1. Run by decile
    2. Run by counting different periods posts
    3. Run by different holding periods
    """
    start = '2015-01-01'
    end = '2019-07-31'
    try:
        modes = sys.argv[1]
        if modes == 'all':
            modes = [f'cmc{idx}_ret' for idx in [1, 3, 5, 10, 15, 20, 30]]
    except IndexError:
        modes = ['cmc10_ret']

    # cs = CrossSignal(start=start, end=end)
    # bs = Backtest(cs.equal_weight_decile_long_short(), start=start, end=end,
    #               path=f'data/params_long_short/')
    # for mode in modes:
    #     interval = int(re.findall('cmc([0-9]+).+', mode)[0])
    #     bs.simulate_one_portfolio(start_date=0, interval=interval)

    for counting in range(1, 11):
        for decile in range(10, 0, -1):
    # for decile in [1]:
    #     for counting in [5]:
            print('*' * 40)
            print(' ' * 9, f'Decile {decile} - Counting {counting}')
            print('*' * 40)
            # decile = round(decile/2, 1)
            if not os.path.exists(f'data/params_top_rank_positive_3pm/Decile {decile} - Counting {counting}'):
                os.mkdir(f'data/params_top_rank_positive_3pm/Decile {decile} - Counting {counting}')

            cs = CrossSignal(start=start, end=end, signal_period=counting, decile=decile, sentiment='positive')

            bs = Backtest(cs.equal_weight_rank_signal(), start=start, end=end,
                          path=f'data/params_top_rank_positive_3pm/Decile {decile} - Counting {counting}')
            for mode in modes:
                interval = int(re.findall('cmc([0-9]+).+', mode)[0])
                bs.simulate_one_portfolio(start_date=0, interval=interval)

if __name__ == '__main__':
    run_backtest()

