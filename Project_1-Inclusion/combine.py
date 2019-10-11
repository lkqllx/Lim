"""This file is designed for pre-processing the data"""

import pandas as pd
import numpy as np

large = pd.read_excel('data/HS_Composite/hs_largecap.xlsx').dropna()
large['Type'] = np.repeat('Large Cap', large.shape[0])
mid = pd.read_excel('data/HS_Composite/hs_midcap.xlsx').dropna()
mid['Type'] = np.repeat('Mid Cap', mid.shape[0])
small = pd.read_excel('data/HS_Composite/hs_smallcap.xlsx').dropna()
small['Type'] = np.repeat('Small Cap', small.shape[0])
combine = pd.concat([large, mid, small], axis=0)
combine.iloc[:, 0] = pd.to_datetime(combine.iloc[:, 0])
combine = combine.sort_values(by=combine.columns[0], ascending=False)
combine.to_csv('data/HS_Composite/combined_history.csv', index=False, encoding='utf_8_sig')