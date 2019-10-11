import pandas as pd
import pandas_datareader as web
import numpy as np
import datetime as dt
import math
import multiprocessing as mp
import itertools
import os


def run(zipped):
    """Function to be run in parallel"""
    annc_date_str, (buy_advance, sell_delay) = zipped
    print(f'Doing -> Buy = {buy_advance} Sell = {sell_delay}')
    stocks = pd.read_csv('data/HS_Composite/remaining_stocks.csv')
    stocks = stocks[stocks['Status'] != 1]  # Remove the already listed stocks

    profit = list()
    annc_date = dt.datetime.strptime(annc_date_str, '%Y-%m-%d')
    start_date = annc_date - dt.timedelta(days=math.ceil(buy_advance / 5 * 7) + 30)  # +30 for requiring enough data
    end_date = annc_date + dt.timedelta(days=math.ceil(sell_delay / 5 * 7) + 30)
    for row in stocks.values:
        ticker = str(row[4]).zfill(4) + '.hk'
        returns = cal_pnl(ticker, annc_date_str, annc_date, buy_advance, sell_delay, start_date, end_date)
        if returns is not None:
            profit.append((ticker, returns))
    stock_ave_pnl = np.round(np.mean([earning for _, earning in profit]), 3)
    benchmark_pnl = cal_pnl('^HSI', annc_date_str, annc_date, buy_advance, sell_delay, start_date, end_date)
    profit.append(('^HSI', benchmark_pnl))
    pd.DataFrame(profit, columns=['Ticker', 'Returns']).\
        to_csv(f'data/pnl/buy_{buy_advance}_sell_{sell_delay}.csv', index=False)
    return [buy_advance, sell_delay, stock_ave_pnl, stock_ave_pnl - benchmark_pnl]


def cal_pnl(ticker, annc_date_str, annc_date, buy_advance, sell_delay, start_date, end_date):
    try:
        if not os.path.exists(f'data/prices/{ticker}.csv'):
            prices = web.get_data_yahoo(ticker, start_date, end_date)
            prices.to_csv(f'data/prices/{ticker}.csv')
        else:
            prices = pd.read_csv(f'data/prices/{ticker}.csv', index_col=0)
            prices.index = pd.to_datetime(prices.index.values)
        prices['Time'] = prices.index.values
        prices.reset_index(inplace=True)
        # Compute the index of announcing date used for relative indexing of buy and sell prices
        target_index = prices.index[prices['Time'] == annc_date].values[0]
        buy_price = prices.Close.loc[target_index - buy_advance]
        sell_price = prices.Close.loc[target_index + sell_delay]
        return round((sell_price - buy_price) / buy_price, 3)
    except Exception as e:
        print(f'{e} - {ticker} - {annc_date_str}')
        return None


def multi_proc(target_dates, buy_period, sell_period):
    pool = mp.Pool(mp.cpu_count())
    profits = pool.map(run, zip(target_dates, itertools.product(buy_period, sell_period)))
    output = pd.DataFrame(profits, columns=['Buy Advance', 'Sell Delay', 'Ave Profit', 'AVE Profit Minus Benchmark'])
    output.to_csv('~/Desktop/performance.csv', index=False)


if __name__ == '__main__':
    # test_length = 10
    # buy_period = list(range(1, test_length))
    # sell_period = list(range(test_length - 1))
    # target_dates = ['2019-09-09'] * (test_length - 1) ** 2
    # multi_proc(target_dates, buy_period, sell_period)

    print(run(['2019-09-09', (5, 0)]))