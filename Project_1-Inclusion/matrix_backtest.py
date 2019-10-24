import pandas as pd
import pandas_datareader as web
import numpy as np
import datetime as dt
import re
import multiprocessing as mp
import os
import itertools

BUY_PERIOD = 14
SELL_PERIOD = 14


def check_fmt(fmt, targets):
    return [corr for corr in targets if re.match(fmt, corr)]


def fetch_price(ticker, date):
    """Retrieve data from yahoo"""
    start = date - dt.timedelta(days=30)
    end = date + dt.timedelta(days=30)
    df = web.get_data_yahoo(ticker + '.hk', start, end)['Close']
    return df


def backtest(data):
    """
    This function is to compute the pnl matrix in terms of different buying and selling period
    :param data: contains (ticker, date, cap, action)
    :param ticker: ticker in "str" format
    :param date: date in "str" format
    :param buy_period: the buying period before announcement in terms of calendar day
    :param sell_peirod: the selling period equal or after announcement in terms of calendar day
    :return: None
    """
    ticker, date, cap, action = data
    fmt = '%Y-%m-%d'
    date = dt.datetime.strptime(date, fmt)
    if os.path.exists(f'data/pnl/{cap}_{action}/{ticker}_{dt.datetime.strftime(date, fmt)}.csv'):
        return
    try:
        df = fetch_price(ticker, date)

        # df.index >= date
        # -> Select the sell prices which is equal or after the announcing date
        # df.index <= (date + dt.timedelta(SELL_PERIOD))
        # -> Select the sell prices #which is equal before annoucing date + SELL_PERIOD
        sell_prices = df[(df.index >= date) & (df.index <= (date + dt.timedelta(SELL_PERIOD)))]

        # df.index < date
        # -> Select the buy prices which is before the announcing date
        # df.index >= （date - dt.timedelta(BUY_PEPRIOD)）
        # -> Select the buy prices which is equal or after annoucing date - BUY_PERIOD
        buy_prices = df[(df.index >= date - dt.timedelta(BUY_PERIOD)) & (df.index < date)]

        buy_prices = buy_prices[::-1]  # inverse the buy prices for better looking

        sell_matrix = np.tile(sell_prices, (buy_prices.shape[0], 1))  # Sell period is 0 - 10 (11 days)
        """
        SAMPLE
        sell_matrix ->
        price at day0, price at day1, ..., price at day10
        price at day0, price at day1, ..., price at day10
        ...
        price at day0, price at day1, ..., price at day10
        price at day0, price at day1, ..., price at day10
        """

        buy_matrix = np.tile(buy_prices.values, (sell_prices.shape[0], 1)).T  # Buy period is 1 - 10 (10 days)
        """
        SAMPLE
        buy_matrix ->
        price at day-1, price at day-1, ..., price at day-1
        price at day-2, price at day-2, ..., price at day-2
        ...
        price at day-9, price at day-9, ..., price at day-9
        price at day-10, price at day-10, ..., price at day-10
        """

        pnl = np.round((sell_matrix - buy_matrix) / buy_matrix, 3)
        pnl = pd.DataFrame(pnl,
                           columns=[f'Sell {num} trading days after announcement'
                                    for num in range(sell_prices.shape[0])],
                           index=[f'Buy {num} trading days before announcement'
                                  for num in range(1, buy_prices.shape[0] + 1)])

        if not os.path.exists(f'data/pnl/{cap}_{action}'):
            os.mkdir(f'data/pnl/{cap}_{action}')
        pnl.to_csv(f'data/pnl/{cap}_{action}/{ticker}_{dt.datetime.strftime(date, fmt)}.csv')
    except Exception as e:
        print(f'{e} - {ticker} - {dt.datetime.strftime(date, fmt)}')


def multi_processes_run(tickers, dates, cap, action):
    pool = mp.Pool(mp.cpu_count())
    pool.map(backtest, zip(tickers, dates, [cap]*len(dates), [action]*len(dates)))


def filter(cap, action, df: pd.DataFrame):
    """
    Filter the wanted DataFrame
    :param cap: capital -> small, mid, large, all
    :param action: action to take -> add, delete
    :param df: the original DataFrame
    :return: filtered DataFrame
    """
    if cap == 'small':
        df = df[df['Type'] == 'Small Cap']
    elif cap == 'mid':
        df = df[df['Type'] == 'Mid Cap']
    elif cap == 'large':
        df = df[df['Type'] == 'Large Cap']

    if action == 'add':
        df = df[df.iloc[:, 2] == r'Add 加入']
    elif action == 'delete':
        df = df[df.iloc[:, 2] == r'Delete 刪除']
    return df


def run():
    """
    history should be in the following format
    Column Name -> Effective Date 生效日期,No. of Constituents,Change 變動,Count 數目,Stock Code 股份代號,Listing Place 上市地點,Stock Name,股份名稱,Type
    Row 1 ->       2019-09-09,110,Add 加入,2.0,788.0,Hong Kong 香港,CHINA TOWER,中國鐵塔,Large Cap
    Row 2 ->       2019-09-09,179,Add 加入,24.0,6055.0,Hong Kong 香港,CTIHK,中煙香港,Small Cap
    """
    history = pd.read_csv('data/HS_Composite/combined_history.csv')

    caps = ['small', 'mid', 'large']  # To decide whether filter the stocks based on their cap (small, mid, large)
    actions = ['add', 'delete']  # To decide addition or deletion
    for cap, action in itertools.product(caps, actions):
        df = filter(cap, action, history)
        tickers = [str(int(ticker)).zfill(4) for ticker in df.iloc[:, 4]]  # Column 4 is the place to store tickers
        tickers = check_fmt(r'\d{4}', tickers)
        dates = [date for date in df.iloc[:, 0]]  # Column 0 is the place to store dates
        dates = check_fmt(r'\d{4}-\d+\d', dates)
        multi_processes_run(tickers, dates, cap, action)
        
        
if __name__ == '__main__':
    """
    This file will generate pnl file for each stock in 'data/HS_Composite/combined_history.csv'.
    The pnl will be saved in data/pnl/%cap%_%action%/%ticker%_%date%.csv
    """
    run()





