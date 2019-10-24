"""
This file will download the prices of target list
"""

import pandas_datareader as web
import pandas as pd
import concurrent.futures
import datetime as dt
from progress.bar import Bar
import threading
import os

lock = threading.RLock()


def prices(targets: list, start='2010-01-01', end='2019-10-15'):
    start = dt.datetime.strptime(start, '%Y-%m-%d')
    end = dt.datetime.strptime(end, '%Y-%m-%d')

    with Bar('Downloading', max=len(targets)) as bar:
        def download(target):
            try:
                if not os.path.exists(f'data/prices/{target}.csv'):
                    df = web.get_data_yahoo(target, start, end)
                    ticker = target.split('.')[0]
                    df.to_csv(f'data/prices/{ticker}.csv')
                    lock.acquire()
                    bar.next()
                    lock.release()
            except Exception as e:
                print(f'Error - {target} - {e} - func prices')
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as pool:
            pool.map(download, targets)


def run_download_prices():
    shanghai_list = pd.read_csv('data/target_list/SH.csv')
    shanghai_list = [str(name) + '.SS' for name in shanghai_list.iloc[:, 0]]
    shenzhen_list = pd.read_csv('data/target_list/SZ.csv')
    shenzhen_list = [str(name).zfill(6) + '.SZ' for name in shenzhen_list.iloc[:, 0]]
    target_list = shenzhen_list + shanghai_list
    prices(target_list)


if __name__ == '__main__':
    run_download_prices()
