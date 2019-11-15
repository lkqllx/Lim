import pandas as pd
import numpy as np
import calendar
import sys
# _tickers = ['600000', '600009']
# for ticker in _tickers:
#     ticker = str(ticker)
#     curr_df = pd.read_csv(f'data/prices/{ticker}.csv', index_col=0, low_memory=False,
#                           names=pd.MultiIndex.from_product([[ticker], ['open', 'close']]))[1:]
#     curr_df.index = pd.to_datetime(curr_df.index)
#     curr_df = curr_df.astype(float)
#     curr_df.loc[:, (ticker, 'cmo_ret')] = (curr_df.loc[:, (ticker, 'close')] - curr_df.loc[:, (ticker, 'open')]) / \
#                                           curr_df.loc[:, (ticker, 'open')]...
#     curr_df.loc[:, (ticker, 'cmc_ret')] = (curr_df.loc[:, (ticker, 'close')] -
#                                            curr_df.loc[:, (ticker, 'close')].shift(1)) / \
#                                           curr_df.loc[:, (ticker, 'close')].shift(1)
#     try:
#         prices_matrix = pd.concat([prices_matrix, curr_df], axis=1)
#     except:
#         prices_matrix = curr_df
# prices_matrix = prices_matrix.iloc[:, prices_matrix.columns.get_level_values(1) == 'cmo_ret']

#
# while True:
#     print('Successful!')
#     print(f'Input - {sys.argv[1]}')

import requests, bs4,itertools
import numpy as np

def get_proxy():
    proxies = []
    url = 'https://free-proxy-list.net/'
    web = requests.get(url)
    soup = bs4.BeautifulSoup(web.content, 'html')
    items = soup.find_all('tr')[1:]
    for item in items:
        cells = item.find_all('td')
        try:
            if cells[6].text == 'yes':
                proxies.append(':'.join([cells[0].text, cells[1].text]))
        except:
            continue
    proxies = np.random.permutation(proxies)
    return itertools.cycle(proxies)

get_proxy()