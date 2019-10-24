import pandas as pd


caps = pd.read_csv('data/HS_Composite/cap.csv')
cap_map = {row[2]: row[3] for row in caps.values}

df = pd.read_csv('data/HS_Composite/combined_history.csv')
df = df[(df.iloc[:, 0] == '2019-09-09') & (df.iloc[:, 2] == 'Add 加入')]
drop_list = []
for idx, row in enumerate(df.values):
    if row[8] == 'Small Cap':
        if cap_map[int(row[4])] <= 5000:
            drop_list.append(idx)
df = df.drop(index=drop_list)
df = df.sort_values(by='Stock Code 股份代號')
df.to_csv('~/Desktop/remaining_stocks.csv', encoding='utf_8_sig', index=False)

