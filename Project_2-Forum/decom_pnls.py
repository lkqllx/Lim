import pandas as pd
import numpy as np

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
        all_percent_df = pd.concat([all_percent_df, pd.Series([ticker_pnl], name=ticker)], axis=1)
    except:
        all_percent_df = pd.Series([ticker_pnl], name=ticker)

all_percent_df.to_csv('C:/Users/andrew.li/Desktop/pnl_contribution.csv')