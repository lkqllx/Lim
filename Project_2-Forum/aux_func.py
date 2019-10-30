# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
import pandas as pd
import datetime as dt
from progress.bar import Bar
# pandas2ri.activate
import numpy as np
import jqdatasdk as jq
import pickle, os, re, calendar


def process_rds_to_csv():
    tickers = pd.read_excel('data/target_list/csi300.xls').iloc[:, 4].values.tolist()
    with Bar('Processing', max=len(tickers)) as bar:
        for ticker in tickers:
            bar.next()
            ticker = str(ticker).zfill(6)
            readRDS = robjects.r['readRDS']
            df = readRDS(f'data/price_shsz/{ticker}.rds')
            df['date'] = df['date'].apply(lambda x: dt.datetime(1970, 1, 1) + dt.timedelta(int(x)))
            df.to_csv(f'data/prices/{ticker}.csv', index=False)


def extract_excess_returns_to_r():
    all_excess = pd.read_csv('R/excess_ret/excess_daily_pnls.csv', index_col=0, parse_dates=True)
    all_excess = all_excess.iloc[:, list(range(1, all_excess.shape[1], 2))]
    all_excess.to_csv(f'R/excess_ret/excess_daily_pnls.csv')
    all_excess['Time'] = all_excess.index.values
    all_excess['Time'] = all_excess['Time'].apply(lambda x: str(x.year))
    targets = ['2015', '2016', '2017', '2018', '2019']
    for target_year in targets:
        curr_df = all_excess[all_excess['Time'] == target_year]
        curr_df.to_csv(f'R/excess_ret/{target_year}.csv')


def decompo_pnls():
    all_pnl = pd.read_csv('C:/Users/andrew.li/Desktop/individual_pnl_cmc10_ret_0.csv', index_col=0, parse_dates=True)
    average_pnl = all_pnl['Total'] + 1
    average_pnl = average_pnl.cumprod()
    final_pnl = average_pnl[-1] - 1
    all_pnl = all_pnl.drop('Total', axis=True)

    all_pnl = all_pnl.apply(lambda x: x / np.count_nonzero(x), axis=1)

    for ticker in all_pnl.columns:
        curr_df = all_pnl.loc[:, ticker] + 1
        curr_df = curr_df.cumprod()
        ticker_pnl = curr_df[-1] - 1

        try:
            all_percent_df = pd.concat([all_percent_df, pd.Series([ticker_pnl/ final_pnl], name=ticker)], axis=1)
        except:
            all_percent_df = pd.Series([ticker_pnl / final_pnl], name=ticker)

    all_percent_df.to_csv('C:/Users/andrew.li/Desktop/pnl_contribution.csv')


def download_members():
    jq.auth('18810906018', '906018')
    all_ticker = []
    dates = [dt.datetime.strptime('2015-01-01', '%Y-%m-%d') + dt.timedelta(31*idx) for idx in range(58)]
    with Bar('Processing', max=len(dates)) as bar:
        for date in dates:
            bar.next()
            curr_tickers = jq.get_index_stocks('000300.XSHG', date)
            all_ticker += curr_tickers
    all_ticker = set(all_ticker)
    with open('data/all_list.pkl', 'wb') as f:
        pickle.dump(all_ticker, f)

def download_prices_from_jq():
    jq.auth('18810906018', '906018')
    with open('data/all_list.pkl', 'rb') as f:
        csi300 = pickle.load(f)
    csi300 = [ticker.split('.')[0] for ticker in csi300]
    existed_files = os.listdir('data/prices')
    existed_files = [file.split('.')[0] for file in existed_files if re.match('.+csv', file)]
    csi300_list = [file for file in csi300 if file not in existed_files]
    with Bar('Downloading', max=len(csi300_list)) as bar:
        for ticker in csi300_list:
            bar.next()
            formated_ticker = jq.normalize_code(ticker)
            curr_prices = jq.get_price(formated_ticker, start_date='2015-01-01', end_date='2019-07-31')
            curr_prices = curr_prices.loc[:, ['open', 'close']]
            curr_prices.columns = ['PX_OPEN', 'PX_LAST']
            curr_prices.to_csv(f'data/prices/{ticker}.csv')


def check_members():
    jq.auth('18810906018', '906018')
    dates = ['2015-01-31', '2016-01-31', '2017-01-31', '2018-01-31', '2019-01-31']
    with Bar('Processing', max=len(dates)) as bar:
        for date in dates:
            bar.next()
            curr_tickers = jq.get_index_stocks('000300.XSHG', date)
            pd.DataFrame(curr_tickers).to_csv(f'~/Desktop/new_result/{date}.csv', index=False)


if __name__ == '__main__':
    # extract_excess_returns_to_r()
    # download_members()
    # download_prices_from_jq()
    check_members()