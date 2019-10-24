import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pandas as pd
import datetime as dt
from progress.bar import Bar
pandas2ri.activate()

tickers = pd.read_excel('data/target_list/csi300.xls').iloc[:, 4].values.tolist()
with Bar('Processing', max=len(tickers)) as bar:
    for ticker in tickers:
        bar.next()
        ticker = str(ticker).zfill(6)
        readRDS = robjects.r['readRDS']
        df = readRDS(f'data/price_shsz/{ticker}.rds')
        df['date'] = df['date'].apply(lambda x: dt.datetime(1970, 1, 1) + dt.timedelta(int(x)))
        df.to_csv(f'data/prices/{ticker}.csv', index=False)