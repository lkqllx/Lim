# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
import pandas as pd
import datetime as dt
from progress.bar import Bar
# pandas2ri.activate
import numpy as np
import jqdatasdk as jq
import pickle, os, re, calendar
from snownlp import SnowNLP
import multiprocessing as mp
import time


def timer(fn):
    """
    Perform as a timer for function
    :param fn: a function object
    :return: a list -> [fn_output, elapsed_time]
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        fn_output = fn(*args, **kwargs)
        end = time.time()
        return [fn_output, int(end - start)]
    return wrapper

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
    # all_excess = pd.read_csv('R/excess_ret/excess_daily_pnls.csv', index_col=0, parse_dates=True)
    # all_excess = all_excess.iloc[:, list(range(1, all_excess.shape[1], 2))]
    all_excess = pd.read_csv('R/excess_ret/original_return.csv', index_col=0, parse_dates=True)
    # all_excess.to_csv(f'R/excess_ret/excess_daily_pnls.csv')
    all_excess['Time'] = all_excess.index.values
    all_excess['Time'] = all_excess['Time'].apply(lambda x: str(x.year))
    targets = ['2015', '2016', '2017', '2018', '2019']
    for target_year in targets:
        curr_df = all_excess[all_excess['Time'] == target_year]
        curr_df.drop('Time', axis=1, inplace=True)
        curr_df.to_csv(f'R/excess_ret/{target_year}.csv')


def decompo_pnls(file, direction='long'):
    # jq.auth('18810906018', '906018')
    all_pnl = pd.read_csv(file, index_col=0, parse_dates=True)
    all_pnl = all_pnl.drop('Total', axis=True)
    all_pnl['Total'] = all_pnl.apply(lambda row: np.mean(row.replace(0, np.nan)), axis=1)
    if direction == 'long':
        average_pnl = all_pnl['Total'] + 1
    else:
        average_pnl = 1 - all_pnl['Total']
    average_pnl = average_pnl.cumprod()
    final_pnl = average_pnl[-1] - 1
    all_pnl = all_pnl.drop('Total', axis=True)

    all_pnl = all_pnl.apply(lambda x: x / np.count_nonzero(x), axis=1)

    for ticker in all_pnl.columns:
        if direction == 'long':
            curr_df = all_pnl.loc[:, ticker]
        else:
            curr_df = - all_pnl.loc[:, ticker]
            curr_df = curr_df * average_pnl.shift(1)
        curr_df = curr_df + 1
        curr_df = curr_df.cumprod()
        ticker_pnl = curr_df[-1] - 1
        try:
            all_percent_df = pd.concat([all_percent_df, pd.Series([ticker_pnl/ final_pnl], name=ticker)], axis=1)
        except:
            all_percent_df = pd.Series([ticker_pnl / final_pnl], name=ticker)
    all_percent_df = all_percent_df.T
    all_percent_df.columns = ['percentage']
    all_percent_df = all_percent_df.sort_values(by='percentage', ascending=False)
    all_percent_df = all_percent_df / all_percent_df.sum()

    valid_tickers = jq.normalize_code(all_percent_df.index.values.tolist())
    q = jq.query(jq.valuation.code, jq.valuation.market_cap).filter(jq.valuation.code.in_(valid_tickers))
    market_cap = jq.get_fundamentals(q, '2019-07-31')
    market_cap.index = market_cap['code'].apply(lambda row: row.split('.')[0])
    market_cap.drop('code', axis=1, inplace=True)
    all_percent_df = pd.concat([all_percent_df, market_cap], axis=1, sort=False)
    all_percent_df.to_csv(f'C:/Users/andrew.li/Desktop/result/large and mid cap/pnl_contribution_15_{direction}.csv')


def download_members():
    # jq.auth('18810906018', '906018')
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
    # jq.auth('18810906018', '906018')
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
            curr_prices = jq.get_price(formated_ticker, start_date='2019-11-01', end_date='2019-12-02')
            curr_prices = curr_prices.loc[:, ['open', 'close']]
            curr_prices.columns = ['PX_OPEN', 'PX_LAST']
            curr_prices.to_csv(f'data/csv_history/prices/{ticker}.csv')


def check_members():
    # jq.auth('18810906018', '906018')
    dates = ['2015-01-31', '2016-01-31', '2017-01-31', '2018-01-31', '2019-01-31']
    with Bar('Processing', max=len(dates)) as bar:
        for date in dates:
            bar.next()
            curr_tickers = jq.get_index_stocks('000300.XSHG', date)
            pd.DataFrame(curr_tickers).to_csv(f'~/Desktop/new_result/{date}.csv', index=False)


def create_caps():
    with open('data/all_list.pkl', 'rb') as f:
        all_list = pickle.load(f)
    all_list = list(all_list)
    all_list = sorted(all_list, key=lambda x: int(x.split('.')[0]))
    with Bar('Downloading', max=len(all_list)) as bar:
        for ticker in all_list:
            bar.next()
            q = jq.query(jq.valuation.circulating_market_cap,jq.valuation.market_cap
                         ).filter(jq.valuation.code.in_([ticker]))
            panel = jq.get_fundamentals_continuously(q, end_date='2019-07-31',
                                                     count=1116)
            try:
                market_caps = pd.concat([market_caps, panel.market_cap], axis=1, sort=True)
                circulating_market_caps = pd.concat([circulating_market_caps, panel.circulating_market_cap], axis=1,
                                                    sort=True)
            except:
                market_caps = panel.market_cap
                circulating_market_caps = panel.circulating_market_cap
    market_caps.columns = [name.split('.')[0] for name in market_caps.columns]
    market_caps.to_csv('data/fundamental/market_caps.csv')
    circulating_market_caps.columns = [name.split('.')[0] for name in circulating_market_caps.columns]
    circulating_market_caps.to_csv('data/fundamental/circulating_market_caps.csv')


def masking_caps():
    market_caps = pd.read_csv('data/fundamental/market_caps.csv', index_col=0, parse_dates=True)
    circulating_market_caps = pd.read_csv('data/fundamental/circulating_market_caps.csv', index_col=0, parse_dates=True)

    effective_dates = ['31-07-2019', '17-06-2019', '17-12-2018', '11-06-2018', '11-12-2017', '12-06-2017',
                       '12-12-2016', '13-06-2016', '30-12-2015', '30-11-2015', '15-06-2015', '14-05-2015',
                       '26-01-2015', '01-01-2015']
    effective_dates = list(map(dt.datetime.strptime, effective_dates, ['%d-%m-%Y'] * len(effective_dates)))
    effective_dates.reverse()

    """This mask matrix will be used to multiply with stocks_post_matrix"""
    mask_matrix = pd.DataFrame(np.nan, index=market_caps.index, columns=market_caps.columns)

    with Bar('Masking', max=len(effective_dates)) as bar:
        for idx, date in enumerate(effective_dates):
            bar.next()
            curr_stocks = jq.get_index_stocks('000300.XSHG', date)
            curr_stocks = [stock.split('.')[0] for stock in curr_stocks]
            if idx != len(effective_dates) - 1:
                """if not last date, we will assign 1 to available stocks until the Previous 
                day of next next effective day"""
                mask_matrix.loc[date:effective_dates[idx + 1] - dt.timedelta(1), curr_stocks] = 1
            else:
                mask_matrix.loc[effective_dates[idx], curr_stocks] = 1
    market_caps = market_caps.fillna(0)
    market_caps = mask_matrix * market_caps
    circulating_market_caps = circulating_market_caps.fillna(0)
    circulating_market_caps = mask_matrix * circulating_market_caps
    market_caps.to_csv('data/fundamental/market_caps.csv')
    circulating_market_caps.to_csv('data/fundamental/circulating_market_caps.csv')

@timer
def run_by_mp(file):
    df = pd.read_csv(f'data/historical/2019-10-15/{file}', low_memory=False)
    df['sentiment'] = 0.5
    for idx, text in enumerate(df['Title'].values):
        try:
            df.loc[idx, 'sentiment'] = SnowNLP(text).sentiments
        except:
            continue
    print(df)
    df.to_csv(f'data/historical/sentiment/{file}', encoding='utf_8_sig', index=False)


def add_sentiment_to_posts():
    files = os.listdir('data/historical/2019-10-15/')
    files = [file for file in files if re.match('.+[csv]', file)]
    pool = mp.Pool(8)
    with Bar('Processing', max=len(files)) as bar:
        for _ in pool.imap_unordered(run_by_mp, files):
            bar.next()


def create_new_benchmark():
    stock_posts = pd.read_csv('data/interim/weekend_stocks_post_matrix_after_masking.csv',
                              index_col=0, parse_dates=True)
    ret_matrix = pd.read_csv('data/interim/ret_matrix.csv',
                             index_col=0, parse_dates=True)

    valid_returns = ret_matrix[stock_posts.notnull()]
    mean_returns = valid_returns.mean(axis=1)
    mean_returns.dropna(inplace=True)
    mean_returns.to_csv('data/interim/equal_weight_benchmark.csv')
    # mean_returns = mean_returns + 1
    # cum_return = mean_returns.cumprod() - 1
    # cum_return.plot()


def add_sentiment_label():
    files = os.listdir('data/historical/sentiment/')
    files = [file for file in files if re.match('.+[csv]', file)]
    with Bar('Labelling', max=len(files)) as bar:
        for file in files:
            bar.next()
            curr_df = pd.read_csv('data/historical/sentiment/' + file, index_col=0, low_memory=False)
            if 'sent_laben' in curr_df.columns:
                curr_df.columns = curr_df.columns[:-1].values.tolist() + ['sent_label']
            else:
                curr_df['sent_label'] = curr_df['sentiment'].apply(lambda x: 0 if x <= 0.5 else 1)
            curr_df.to_csv('data/historical/sentiment/' + file, encoding='utf_8_sig')


def download_current_universe_price(curr_date):
    tickers = jq.get_index_stocks('000300.XSHG')
    with Bar('Downloading prices', max=len(tickers)) as bar:
        for ticker in tickers:
            bar.next()
            curr_price = jq.get_price(ticker, start_date='2019-11-01', end_date=curr_date)
            name = ticker.split('.')[0]
            curr_price.to_csv(f'csv_history/prices/{name}.csv')

    files = os.listdir('csv_history/prices')
    files = [file for file in files if 'csv' in file]
    with Bar('Downloading prices', max=len(files)) as bar:
        for file in files:
            bar.next()
            curr_df = pd.read_csv('csv_history/prices/' + file, index_col=0).close
            curr_df.name = file.split('.')[0]
            try:
                all_df = pd.concat([all_df, curr_df], axis=1)
            except:
                all_df = curr_df
    ret_matrix = (all_df.shift(-1) - all_df) / all_df
    ret_matrix.dropna(inplace=True)
    ret_matrix.to_csv('csv_history/ret_matrix.csv')

    csi300 = jq.get_price('000300.XSHG', start_date='2019-11-01', end_date=curr_date).close
    csi300 = (csi300.shift(-1) - csi300) / csi300
    csi300.to_csv('csv_history/csi300.csv')

def quick_backtest(curr_date):
    files = os.listdir('csv_history/')
    files = [file for file in files if re.match('table_230.+.csv', file)]
    targets = []
    for file in files:
        curr_df = pd.read_csv('csv_history/' + file, index_col=0)
        date = curr_df['Date'][0]
        curr_signal = curr_df.index[curr_df['Rank_neg_8'] == 10]
        curr_signal = [signal.split(' ')[0] for signal in curr_signal]
        targets.append((date, curr_signal))

    csi300 = pd.read_csv('csv_history/csi300.csv', index_col=0, names=['csi300'])
    ret_matrix = pd.read_csv('csv_history/ret_matrix.csv', index_col=0)
    cum_ret = 1
    rets = []
    targets = sorted(targets, key=lambda x: dt.datetime.strptime(x[0], '%Y-%m-%d'))
    for combo in targets:
        try:
            if combo[0] == curr_date:
                continue
            curr_ret = ret_matrix.loc[combo]
            ave_ret = curr_ret.mean()
            rets.append((combo[0], round(ave_ret - csi300.loc[combo[0], 'csi300'], 4)))
            cum_ret = cum_ret * (1 + ave_ret - csi300.loc[combo[0], 'csi300'])
        except:
            continue
    print(rets)
    print(cum_ret)


if __name__ == '__main__':
    jq.auth('18810906018', '906018')
    # extract_excess_returns_to_r()
    # download_members()

    # check_members()
    # decompo_pnls(f'~/Desktop/result/large and mid cap/individual_pnl_cmc15_ret_0.csv', 'long')
    # create_caps()
    # masking_caps()
    # add_sentiment_to_posts()
    # create_new_benchmark()
    # add_sentiment_label()
    # (_, time_used) = run_by_mp('000002.csv')
    # print(time_used)
    date = '2019-12-05'
    # download_current_universe_price(date)
    quick_backtest(date)