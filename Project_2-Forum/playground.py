import pandas as pd
import numpy as np
import calendar
# _tickers = ['600000', '600009']
# for ticker in _tickers:
#     ticker = str(ticker)
#     curr_df = pd.read_csv(f'data/prices/{ticker}.csv', index_col=0, low_memory=False,
#                           names=pd.MultiIndex.from_product([[ticker], ['open', 'close']]))[1:]
#     curr_df.index = pd.to_datetime(curr_df.index)
#     curr_df = curr_df.astype(float)
#     curr_df.loc[:, (ticker, 'cmo_ret')] = (curr_df.loc[:, (ticker, 'close')] - curr_df.loc[:, (ticker, 'open')]) / \
#                                           curr_df.loc[:, (ticker, 'open')]
#     curr_df.loc[:, (ticker, 'cmc_ret')] = (curr_df.loc[:, (ticker, 'close')] -
#                                            curr_df.loc[:, (ticker, 'close')].shift(1)) / \
#                                           curr_df.loc[:, (ticker, 'close')].shift(1)
#     try:
#         prices_matrix = pd.concat([prices_matrix, curr_df], axis=1)
#     except:
#         prices_matrix = curr_df
# prices_matrix = prices_matrix.iloc[:, prices_matrix.columns.get_level_values(1) == 'cmo_ret']



