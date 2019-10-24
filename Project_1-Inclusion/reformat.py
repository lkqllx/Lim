import pandas_datareader as web
import pandas as pd
import os
import datetime as dt
import math

def cal_return(ticker, start, end):
    start_date = start - dt.timedelta(30)
    end_date = end + dt.timedelta(30)
    ticker = str(ticker).zfill(4)
    if not os.path.exists(f'data/prices/{ticker}.hk.csv'):
        price = web.get_data_yahoo(ticker + '.hk', start_date, end_date)
        price.to_csv(f'data/prices/{ticker}.hk.csv')
    else:
        price = pd.read_csv(f'data/prices/{ticker}.hk.csv', index_col=0)
        price.index = pd.to_datetime(price.index.values)
    price = price.Close
    ret = (price - price.shift(1)) / price.shift(1)
    ret.dropna(inplace=True)
    ret.name = ticker
    ret.fillna(0, inplace=True)
    return ret.round(3)


def run(annc_date_str, period):
    stocks = pd.read_csv(f'data/HS_Composite/remaining_stocks.csv')
    caps = pd.read_csv(f'data/HS_Composite/cap.csv')
    caps_dict = {row[2]: row[3] for row in caps.values}
    stocks['Cap'] = stocks.apply(lambda x: int(caps_dict[x[4]]) if x[8] == 'Small Cap' else None, axis=1)

    annc_date = dt.datetime.strptime(annc_date_str, '%Y-%m-%d')
    start_date = annc_date - dt.timedelta(days=math.ceil(period / 5 * 7))
    end_date = annc_date + dt.timedelta(days=math.ceil(period / 5 * 7))
    tickers = []
    for row in stocks.values:
        ticker = row[4]
        status = row[9]
        curr_returns = cal_return(ticker, start_date, end_date)
        # if status == 1:
        #     curr_returns = pd.Series([None] * curr_returns.size, name=ticker, index=curr_returns.index)
        tickers.append(ticker)
        try:
            all_returns = pd.concat([all_returns, curr_returns[(curr_returns.index >= start_date)
                                                               & (curr_returns.index <= end_date)]], sort=False, axis=1)
        except UnboundLocalError:
            all_returns = curr_returns[(curr_returns.index >= start_date) & (curr_returns.index <= end_date)]
    all_returns.columns = tickers
    all_returns = all_returns.T
    all_returns.columns = ['Ret - ' + dt.datetime.strftime(date, '%Y-%m-%d') for date in all_returns.columns]
    stocks.index = stocks['Stock Code 股份代號']
    stocks = pd.concat([stocks, all_returns], axis=1, sort=False)
    stocks.to_csv('~/Desktop/concat_info.csv', index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    run('2019-09-09', 10)
