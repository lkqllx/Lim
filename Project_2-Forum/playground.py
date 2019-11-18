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

<<<<<<< HEAD
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
=======
# while True:
#     print('Successful!')
#     print(f'Input - {sys.argv[1]}')

from openpyxl import Workbook
from openpyxl.styles import colors
from openpyxl.formatting.rule import DataBarRule
curr_list = [['li', 1],
             ['li2', 2],
             ['li3', 3],
             ['li4', 4],
             ['li5', 5],
             ['li6', 6]]
filename = 'text.xlsx'
workbook = Workbook()
new_sheet = workbook.active
new_sheet['A1'] = 'Ticker'
new_sheet['B1'] = 'Posts'
max_val = 0
for row_idx, row in enumerate(range(2, 2 + len(curr_list))):
    for col_idx, col in enumerate(['A', 'B']):
        if col == 'B':
            new_sheet[f'{col}{row}'] = int(curr_list[row_idx][col_idx])
            if int(curr_list[row_idx][col_idx]) > max_val:
                max_val = int(curr_list[row_idx][col_idx])
        else:
            new_sheet[f'{col}{row}'] = curr_list[row_idx][col_idx]
data_bar_rule = DataBarRule(start_type="num",
                            start_value=0,
                            end_type="num",
                            end_value=max_val,
                            color="0e71c7")
new_sheet.conditional_formatting.add(f"B2:B{2 + len(curr_list)}", data_bar_rule)
workbook.save(filename)
>>>>>>> 310cc3af5f9ba82b8d66998ab4763dc0c85d61e2
