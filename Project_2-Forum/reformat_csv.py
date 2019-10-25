import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pandas as pd
import datetime as dt
from progress.bar import Bar
pandas2ri.activate()


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


if __name__ == '__main__':
    extract_excess_returns_to_r()