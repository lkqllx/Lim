import bs4
import numpy as np
import pandas as pd
import requests
import datetime as dt
import functools
import time


def timer(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        fn_output = fn(*args, **kwargs)
        end = time.time()
        return [fn_output, float(end - start)]
    return wrapper


class Stock:
    """
    Fetch data from http://www.eastmoney.com
    SAMPLE columns:
        -> number of read
        -> comments
        -> title
        -> author
        -> issued time
    """

    def __init__(self, ticker):
        self._ticker = ticker
        self.info_list = []
        self.time_parsing = 0
        self.time_req = 0

    @staticmethod
    @timer
    def req_page(web: str):
        page = requests.get(web)
        return page

    @timer
    def parsing(self, page):
        soup = bs4.BeautifulSoup(page.content, 'html.parser')
        target_list = soup.find_all(class_='articleh normal_post')
        if not target_list:
            """If no target list, terminate the program"""
            return False

        for each_div in target_list:
            try:
                each_span = each_div.find_all('span')
                # TODO - Some problems about the comparison
                item_type = each_span[2].find('em').contents[0] if each_span[2].find('em').contents[0] is None else 'target'
                self.info_list.append([each_span[4].contents[0], each_span[2].find('a').contents[0], item_type,
                                       each_span[0].contents[0], each_span[1].contents[0],each_span[3].find('font').contents[0]])
            except:
                pass
        return True

    def run(self):
        count = 1
        working = True
        while working:
            web = f'http://guba.eastmoney.com/list,{self._ticker},f_{count}.html'
            page, time_req = self.req_page(web)
            self.time_req += time_req
            if not page:
                """END"""
                break
            working, time_parsing = self.parsing(page)
            self.time_parsing += time_parsing
            count += 1
            if count % 20 == 0:
                print('Current count {}'.format(count))
                print('Time Req - {}'.format(round(self.time_req, 3)))
                print('Time Parsing - {}'.format(round(self.time_parsing, 3)))


def reformat_date(df: pd.DataFrame):
    curr_year = dt.datetime.now().year
    df['Time'] = df['Time'].apply(lambda x: '{}-'.format(curr_year) + x)
    df['Time'] = pd.to_datetime(df['Time'])
    df['cmp'] = np.repeat(False, df.shape[0])
    while not np.all(df['cmp']):
        """Exit when all(df['cmp']) = True"""
        df['diff'] = df['Time'] - df['Time'].shift(1)
    print(df.index)


if __name__ == '__main__':
    tickers = ['000001', 'hk00005']
    for ticker in tickers:
        print(f'Doing {ticker}')
        stock = Stock(ticker)
        stock.run()
        df = pd.DataFrame(stock.info_list, columns=['Time', 'Title', 'Types', 'Number of reads', 'Comments',
                                                    'Author'])
        df.to_csv(f'data/{ticker}.csv', encoding='utf_8_sig', index=False)
    df = pd.read_csv('data/hk00001.csv')
    reformat_date(df)

