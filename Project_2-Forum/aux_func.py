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
        curr_df.drop('Time', axis=1)
        curr_df.to_csv(f'R/excess_ret/{target_year}.csv')


def decompo_pnls(direction='long'):
    # jq.auth('18810906018', '906018')
    for idx in [10]:
        all_pnl = pd.read_csv(f'~/Desktop/decomp/'
                              f'individual_pnl_cmc{idx}_ret_0.csv', index_col=0, parse_dates=True)
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

        valid_tickers = jq.normalize_code(all_percent_df.index.values.tolist())
        q = jq.query(jq.valuation.code, jq.valuation.market_cap).filter(jq.valuation.code.in_(valid_tickers))
        market_cap = jq.get_fundamentals(q, '2019-07-31')
        market_cap.index = market_cap['code'].apply(lambda row: row.split('.')[0])
        market_cap.drop('code', axis=1, inplace=True)
        all_percent_df = pd.concat([all_percent_df, market_cap], axis=1, sort=False)
        all_percent_df.to_csv(f'C:/Users/andrew.li/Desktop/decomp/pnl_contribution_{idx}.csv')


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
            curr_prices = jq.get_price(formated_ticker, start_date='2015-01-01', end_date='2019-07-31')
            curr_prices = curr_prices.loc[:, ['open', 'close']]
            curr_prices.columns = ['PX_OPEN', 'PX_LAST']
            curr_prices.to_csv(f'data/prices/{ticker}.csv')


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

if __name__ == '__main__':
    jq.auth('18810906018', '906018')
    # extract_excess_returns_to_r()
    # download_members()
    # download_prices_from_jq()
    # check_members()
    # decompo_pnls('short')
    # create_caps()
    masking_caps()
