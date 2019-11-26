import bs4
import numpy as np
import pandas as pd
import datetime as dt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
import concurrent.futures
import requests
import threading
import time
import re, pickle, itertools
import multiprocessing as mp
import jqdatasdk as jq
from progress.bar import Bar
import logging
from openpyxl import Workbook
from snownlp import SnowNLP
from openpyxl.styles import colors
from openpyxl.formatting.rule import DataBarRule

logging.basicConfig(filename=f'logs/daily_routine_{dt.datetime.now().year}{dt.datetime.now().month}'
                             f'{dt.datetime.now().day}_{dt.datetime.now().hour}.log',
                    filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.ERROR)
lock = threading.RLock()
processer_lock = mp.Lock()
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument('--log-level=3')
chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
chrome_options.add_argument('--no-proxy-server')
chrome_options.add_argument("--proxy-server='direct://'")
chrome_options.add_argument("--proxy-bypass-list=*")

# def get_proxy():
#     proxies = []
#     url = 'https://free-proxy-list.net/'
#     web = requests.get(url)
#     soup = bs4.BeautifulSoup(web.content, 'lxml')
#     items = soup.find_all('tr')[1:]
#     for item in items:
#         cells = item.find_all('td')
#         try:
#             if cells[6].text == 'yes':
#                 proxies.append(':'.join([cells[0].text, cells[1].text]))
#         except:
#             continue
#     proxies = np.random.permutation(proxies)
#     return itertools.cycle(proxies)
# proxies = get_proxy()

def timer(fn):
    """
    Perform as a timer for function
    :param fn: a function object
    :return: a list -> [fn_output, elapsed_time]
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        fn_output = fn(*args, **kwargs)
        end = time.time()
        return [fn_output, int(end - start)]
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

    def __init__(self, ticker, num_pages):
        self._ticker = ticker
        self.info_list = []
        self.time_req = 0
        self.total_pages = 1
        self.all_websites = {}
        self.thread_local = threading.local()
        self.num_pages = num_pages

    @staticmethod
    @timer
    def req_page(web: str):
        page = requests.get(web)
        return page

    @timer
    def parsing(self, page, web_base='http://guba.eastmoney.com'):
        """
        Perform parsing function
        :param page: the string format page source from requires
        :param web_base: the base link
        :return: None
        """
        soup = bs4.BeautifulSoup(page, 'html.parser')
        target_list = soup.find_all(class_='articleh normal_post')
        for target in target_list:
            try:
                links = target.find('span', class_='l3 a3').find_all('a')
                for link in links:
                    if re.match('/news,[\w\d]+,[\w\d]+.html', link.get('href')):
                        break
                text = target.text.strip('\n')
                # readings, comments, title, author, published_time = text.split('\n')

                all_info = text.split('\n')
                if not re.match('[\d]+-[\d]+.+', all_info[-1]):
                    continue
                published_time = all_info[-1]
                title = ''.join(all_info[2:-2])
                self.info_list.append((published_time, title,  web_base+link.get('href')))
            except Exception as e:
                # print(f'Parsing - {e}')
                logging.error(f'Parsing - {e}', exc_info=True)
                return False
        return True

    def call_webdriver(self):
        first_page = f'http://guba.eastmoney.com/list,{self._ticker}.html'  # Use selenium to get the total pages
        try:
            with webdriver.Chrome('./chromedriver', options=chrome_options) as driver:
                driver.set_page_load_timeout(15)
                driver.get(first_page)
                soup = bs4.BeautifulSoup(driver.page_source, 'html.parser')
                elements = soup.find_all('span', class_='sumpage')
                self.num_pages = min(int(elements[0].text), self.num_pages)
                driver.close()
                driver.quit()
                return True
        except Exception as e:
            logging.error(f'Error - {self._ticker} - {str(e)} - Stock.run', exc_info=True)
            return False

    def run(self):
        """
        For the stock, function will require all the pages and parse the website
        :return: True/False indicating that whether the stock is acquired successfully or not
        """
        start = time.time()
        new_thread = threading.Thread(target=self.call_webdriver, daemon=True)
        new_thread.start()

        """
        Download all the websites and join them into a single string variable (all_in_one) for parsing
        """

        sites = [f'http://guba.eastmoney.com/list,{self._ticker},f_{count}.html'
                 for count in range(self.num_pages) if count != 1]

        self.req_flag = []
        tries = 5
        while ((not self.req_flag) or (not all(self.req_flag))) and tries >= 0:
            self.req_flag = []
            tries -= 1
            self.download_all_sites(sites)
        time_web = int(time.time() - start)
        time_parsing = time.time()
        all_sites = sorted(self.all_websites.items(), key=lambda x: int(x[0]))
        if len(all_sites) != len(sites):
            """Make sure that all sites are downloaded successfully"""
            return False, 0, 0

        for _, page in all_sites:
            successful, time_elapsed = self.parsing(page)
            time_parsing += time_elapsed
            if not successful:
                return False, 0, 0

        time_parsing = int(time.time() - time_parsing)
        return True, time_parsing, time_web

    def run_by_one_page(self):
        """
        For the stock, function will require all the pages and parse the website
        :return: True/False indicating that whether the stock is acquired successfully or not
        """
        start = time.time()
        new_thread = threading.Thread(target=self.call_webdriver, daemon=True)
        new_thread.start()

        """
        Download all the websites and join them into a single string variable (all_in_one) for parsing
        """


        max_pages = 20
        download_complete = False
        count = 0

        dates = os.listdir(f'//fileserver01/limdata/data/individual staff folders/andrew li/daily/{self._ticker}')
        dates = [dt.datetime.strptime(date.split('.')[0], '%Y-%m-%d') for date in dates
                 if re.match('[\d]+-[\d]+-[\d]+.csv', date)]
        max_date = max(dates).strftime('%Y-%m-%d')
        df = pd.read_csv(f'//fileserver01/limdata/data/individual staff folders/andrew li/daily'
                         f'/{self._ticker}/{max_date}.csv', index_col=0, parse_dates=True)
        latest_data = df.index[0]
        while (not download_complete) and (max_pages >= 1):
            if count == 1:
                count += 1
                continue
            sites = [f'http://guba.eastmoney.com/list,{self._ticker},f_{count}.html']

            self.req_flag = []
            tries = 5
            while ((not self.req_flag) or (not all(self.req_flag))) and tries >= 0:
                self.req_flag = []
                tries -= 1
                self.download_all_sites(sites)
            time_web = int(time.time() - start)
            time_parsing = time.time()
            page = self.all_websites[f'{count}']
            successful, time_elapsed = self.parsing(page)
            if not successful:
                return False, 0, 0

            curr_page_time = dt.datetime.strptime(str(dt.datetime.now().year) +
                                                  '-' + self.info_list[-1][0], '%Y-%m-%d %H:%M')
            if curr_page_time < latest_data:
                """If the lastest time in DB is larger than the time in the current page,
                we do not need to download further pages"""
                download_complete = True

            count += 1

        time_parsing = int(time.time() - time_parsing)
        return True, time_parsing, time_web

    def get_session(self):
        if not hasattr(self.thread_local, "session"):
            self.thread_local.session = requests.Session()
        return self.thread_local.session

    def download_site(self, url):
        """
        Make sure the bar.next() will not be printed by different threads in the same time
        :param url: the url link to be scraped
        """
        # lock.acquire()
        # bar.next()
        # lock.release()
        session = self.get_session()
        try:
            # with session.request(method='GET', url=url, timeout=30, proxies={'http': next(proxies),
            #                                                                  'https': next(proxies)}) as response:
            with session.request(method='GET', url=url, timeout=30) as response:
                try:
                    count = re.findall('list,[\w\d]+,f_(\d+).html', url)[0]
                except Exception as e:
                    count = url
                lock.acquire()
                self.all_websites[count] = response.text
                lock.release()
                self.req_flag.append(True)
        except Exception as e:
            # print(f'Timeout - {url}').
            logging.exception(f'Timeout - {url}')
            self.req_flag.append(False)
            time.sleep(10)

    def download_all_sites(self, sites):
        """
        Download all the pages to all_websites
        :param sites: the total sites to be scraped
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(self.download_site, sites)

    def reformat_date(self, df: pd.DataFrame):
        """
        Since there is no year information in the dataset, we need to add this info
        Currently implementation is to compare the two dates. If the later one is greater than
        the previous one, the year will be reduce one.
        :param df: pd.DataFrame which contains the scraped info
        :return: pd.DataFrame which add year info to the time
        """

        df['Time_to_cmp'] = df['Datetime'].apply(lambda x: '{}-'.format(2000) + x)
        df['Time_to_cmp'] = pd.to_datetime(df['Time_to_cmp'], format='%Y-%m-%d %H:%M')
        df['diff'] = df['Time_to_cmp'] - df['Time_to_cmp'].shift(1)

        """
        Compare the current date with previous one
        If timedelta > 0 -> There might be a chance the year change
        If timedelta <= 0 -> The data is fine  (But there might be a marginal case that
        Like three consecutive days
        2018-09-09, 2017-09-07, 2016-10-06 will be re-formatted to
        2018-09-09, 2018-09-07, 2016-10-06
        """
        df['cmp'] = np.where(df['diff'] <= dt.timedelta(0), True, False)
        indexs = df.index[df['cmp'] == False].values.tolist()
        links = []
        links_indexs = []
        for idx in indexs:
            if not re.match('.+qa_list.aspx', df.loc[idx, 'Link']):
                if not re.match('http.+http.+', df.loc[idx, 'Link']):
                    links.append(df.loc[idx, 'Link'])
                    links_indexs.append(idx)

        """
        Scrape the year info given the links
        """
        # global all_websites
        # global bar
        self.all_websites = {}
        # bar = Bar('Formatting', max=len(links))
        target_years = []
        self.req_flag = []
        tries = 5
        while ((not self.req_flag) or (not all(self.req_flag))) and tries >= 0:
            self.req_flag = []
            self.all_websites = {}
            tries -= 1
            self.download_all_sites(links)


        # bar.finish()

        try:
            """To handle the marginal case that two links are the same"""
            sorted_all_websites = [self.all_websites[link] for link in links]
        except Exception as e:
            # print(f'Error - {str(e)} - links_reformatting')
            logging.error(f'Error - {str(e)} - links_reformatting', exc_info=True)
            return

        for page in sorted_all_websites:
            try:
                soup = bs4.BeautifulSoup(page, 'html.parser')
                text = soup.find('div', class_='zwfbtime').text
                year = re.findall('([\d]+)-[\d]+-[\d]+', text)[0]
                target_years.append(year)
            except Exception as e:
                # print(f'Error - {str(e)} - reformat_date - find_year_loop')
                logging.error(f'Error - {str(e)} - reformat_date - find_year_loop', exc_info=True)
                return
        """
        Reformat the original DataFrame
        """
        for idx in range(len(links_indexs)):
            try:
                if idx == len(links_indexs) - 1:
                    """Last index"""
                    df.loc[links_indexs[idx]:, 'Datetime'] = \
                        df.loc[links_indexs[idx]:, 'Datetime'].apply(lambda x: '{}-'.format(target_years[idx]) + x)
                else:
                    """Not last index, we will add year info between links_indexs[idx]:links_indexs[idx+1]"""
                    df.loc[links_indexs[idx]:links_indexs[idx+1]-1, 'Datetime'] =  \
                        df.loc[links_indexs[idx]:links_indexs[idx+1]-1, 'Datetime'].apply(lambda x: '{}-'.format(target_years[idx]) + x)
            except Exception as e:
                logging.error(f'Error - {str(e)} - reformat_date - reformat_loop', exc_info=True)
                return
        return df


def run_update_historical_data(args):
    """
    Scrape the stock info given its ticker
    :param ticker: the stock to be scraped
    :return: None
    """
    # print('-' * 20, f'Doing {ticker}', '-' * 20)
    ticker, num_pages = args
    complete = False
    max_epoch = 5
    while (not complete) and (max_epoch >= 0):
        max_epoch -= 1
        try:
            stock = Stock(ticker, num_pages)
            if num_pages != -1:
                complete, time_parsing, time_web = stock.run()
            else:
                complete, time_parsing, time_web = stock.run_by_one_page()
            if complete:
                df = pd.DataFrame(stock.info_list, columns=['Datetime', 'Title', 'Link'])
                formated_df = stock.reformat_date(df)
                formated_df.drop(['Time_to_cmp', 'diff', 'cmp'], axis=1, inplace=True)
                formated_df.loc[:, 'Datetime'] = formated_df['Datetime'].apply(
                    lambda row: dt.datetime.strptime(row, '%Y-%m-%d %H:%M'))
                formated_df.loc[:, 'Date'] = formated_df['Datetime'].apply(
                    lambda row: dt.datetime.strftime(row, '%Y-%m-%d'))
                formated_df.loc[:, 'Time'] = formated_df['Datetime'].apply(
                    lambda row: dt.datetime.strftime(row, '%H:%M:%S'))

                if not os.path.exists(f'//fileserver01/limdata/data/individual staff folders/andrew li/daily/{ticker}'):
                    os.mkdir(f'//fileserver01/limdata/data/individual staff folders/andrew li/daily/{ticker}')

                all_existed_date = os.listdir(f'//fileserver01/limdata/data/individual staff folders/andrew li/'
                                              f'daily/{ticker}')
                all_existed_date = [date.split('.')[0] for date in all_existed_date
                                    if re.match('[\d]+-[\d]+-[\d]+.csv', date)]
                if all_existed_date:
                    """Update largest date in the folder"""
                    max_date = max([dt.datetime.strptime(date, '%Y-%m-%d') for date in all_existed_date])
                    max_date = max_date.strftime('%Y-%m-%d')
                    prev_df = pd.read_csv('//fileserver01/limdata/data/individual staff folders/andrew li/daily/'
                                          '{}/{}.csv'.format(ticker, max_date),
                                          index_col=0, parse_dates=True)
                    filtered_df = formated_df[formated_df['Date'] == max_date]
                    filtered_df.set_index('Datetime', inplace=True)
                    filtered_df = filtered_df[(filtered_df.index > prev_df.index[0])]
                    filtered_df.loc[:, 'Sentiment']  = \
                        filtered_df['Title'].apply(lambda x: 'Positive' if SnowNLP(x).sentiments >= 0.5 else 'Negative')
                    prev_df = pd.concat([filtered_df, prev_df], sort=True)
                    prev_df.to_csv(f'//fileserver01/limdata/data/individual staff folders/andrew li/'
                                   f'daily/{ticker}/{max_date}.csv', encoding='utf_8_sig')

                date_labels = np.unique(formated_df['Date']).tolist()
                for date in date_labels:
                    if not date in all_existed_date:
                        filtered_df = formated_df[formated_df['Date'] == date]
                        filtered_df.loc[:, 'Sentiment'] = \
                            filtered_df['Title'].apply(lambda x: 'Positive' if SnowNLP(x).sentiments >= 0.5 else 'Negative')
                        filtered_df.to_csv(f'//fileserver01/limdata/data/individual staff folders/andrew li/daily'
                                           f'/{ticker}/{date}.csv', index=False, encoding='utf_8_sig')

                return ticker, True, time_parsing, time_web
                # else:
                #     formated_df.set_index('Datetime', inplace=True)
                #     prev_df = pd.read_csv(f'data/daily/{ticker}.csv', index_col=0, parse_dates=True)
                #     filtered_df = formated_df[(formated_df.index > prev_df.index[0])]
                #     filtered_df.loc[:, 'Sentiment']  = filtered_df['Title'].apply(lambda x: 'Positive' if SnowNLP(x).sentiments
                #                                                           >= 0.5 else 'Negative')
                #     prev_df = pd.concat([filtered_df, prev_df], sort=True)
                #     prev_df.to_csv(f'data/daily/{ticker}.csv', encoding='utf_8_sig')
                #     return ticker, True, time_parsing, time_web
            else:
                logging.info(f'Missing the Value of {ticker}')

        except Exception as e:
            max_epoch -= 1
            logging.error(f'Error - {ticker} - {str(e)} - run_update_historical_data', exc_info=True)
            complete = False
            continue
    return ticker, False, 0, 0


@timer
def run_by_historical_multiprocesses(csi300, num_pages, num_cores):
    """
    Multiprocess function to speed up the program
    :return: None
    """
    results = []
    time_parsing = 0
    time_web = 0
    with Bar('Downloading', max=len(csi300)) as bar:
        with mp.Pool(num_cores) as pool:
            for output in pool.imap_unordered(run_update_historical_data, zip(csi300, [num_pages] * len(csi300))):
                bar.next()
                time_parsing += output[2]
                time_web += output[3]
                results.append(output[:2])
    return results, time_web, time_parsing


def write_into_excel(curr_list):
    filename = 'data/daily_summary.xlsx'
    workbook = Workbook()
    sheet = workbook.active
    sheet['A1'] = 'Ticker'
    sheet['B1'] = 'Posts'
    max_val = 0
    for row_idx, row in enumerate(range(2, 2 + len(curr_list))):
        for col_idx, col in enumerate(['A', 'B']):
            if col == 'B':
                sheet[f'{col}{row}'] = int(curr_list[row_idx][col_idx])
                if int(curr_list[row_idx][col_idx]) > max_val:
                    max_val = int(curr_list[row_idx][col_idx])
            else:
                sheet[f'{col}{row}'] = curr_list[row_idx][col_idx]
    data_bar_rule = DataBarRule(start_type="num",
                                start_value=0,
                                end_type="num",
                                end_value=max_val,
                                color="0e71c7")
    sheet.conditional_formatting.add(f"B2:B{2 + len(curr_list)}", data_bar_rule)
    workbook.save(filename)


def update(num_pages, num_cores=4):
    if not os.path.exists('data/current_list.pkl'):
        jq.auth('18810906018', '906018')
        csi300 = jq.get_index_stocks('000300.XSHG', dt.datetime.strftime(dt.datetime.now(), format='%Y-%m-%d'))
        with open('data/current_list.pkl', 'wb') as f:
            pickle.dump(csi300, f)
    else:
        with open('data/current_list.pkl', 'rb') as f:
            csi300 = pickle.load(f)
    csi300 = [ticker.split('.')[0] for ticker in csi300][:300]

    """Number of wanted pages"""
    (curr_list, time_web, time_parsing), time_used = run_by_historical_multiprocesses(csi300, num_pages, num_cores)
    print(f'Time Elapsed - {time_used}')
    print(f'Time Web - {time_web}')
    print(f'Time Parse - {time_parsing}')

    """If there is any failure case, we will exclude those tickers
    and re-run the downloading program until all True"""
    all_succesful = all(np.not_equal(list(zip(*curr_list))[1], False))
    succesful_list = [(ticker, download_flag) for ticker, download_flag in curr_list if download_flag != False]
    not_succesful_list = [ticker for ticker, download_flag in curr_list if download_flag == False]
    count = 5
    while (not all_succesful) and count >= 0:
        try:
            print('Failed Ticker', f'\n, '.join(not_succesful_list))
            (new_curr_list, time_web, time_parsing), time_used = run_by_historical_multiprocesses(not_succesful_list,
                                                                                                  num_pages, num_cores)
            new_succesful_list = [(ticker, download_flag) for ticker, download_flag in new_curr_list if
                                  download_flag != False]
            succesful_list = succesful_list + new_succesful_list

            """Update not_succesful_list for next epoch"""
            not_succesful_list = [ticker for ticker, download_flag in new_curr_list if download_flag == False]
            all_succesful = all(np.not_equal(list(zip(*new_curr_list))[1], False))
            count -= 1
        except:
            count -= 1


def create_current_summary_table(start: dt.datetime, end: dt.datetime, _time: str):
    files = os.listdir('//fileserver01/limdata/data/individual staff folders/andrew li/daily')
    tickers = [file for file in files if re.match('[\d]+', file)]
    info_list = []
    date_range = pd.date_range(start= start, end=end, normalize=True)
    date_range = [date.strftime('%Y-%m-%d') for date in date_range]
    with Bar('Creating Table', max=len(files)) as bar:
        for ticker in tickers:
            for date in date_range:
                try:
                    try:
                        curr_date_df = pd.read_csv(f'//fileserver01/limdata/data/individual staff folders/'
                                                   f'andrew li/daily/{ticker}/{date}.csv', index_col=0, parse_dates=True)
                        curr_ticker = pd.concat([curr_date_df, curr_ticker], sort=True)
                    except UnboundLocalError:
                        curr_ticker = pd.read_csv(f'//fileserver01/limdata/data/individual staff folders/'
                                                   f'andrew li/daily/{ticker}/{date}.csv', index_col=0, parse_dates=True)
                except:
                    logging.exception(f'Out-of-Range {date}-{ticker}')
                    continue
            lookback_1 = curr_ticker[(curr_ticker.index >= start + dt.timedelta(9)) & (curr_ticker.index < end)]
            lookback_8 = curr_ticker[(curr_ticker.index >= start + dt.timedelta(2)) & (curr_ticker.index < end)]
            lookback_10 = curr_ticker[(curr_ticker.index >= start) & (curr_ticker.index < end)]
            num_posts_pos = lookback_1[lookback_1['Sentiment'] == 'Positive']['Sentiment'].count()
            num_posts_neg = lookback_1[lookback_1['Sentiment'] == 'Negative']['Sentiment'].count()
            num_posts_all = num_posts_pos + num_posts_neg

            num_posts_neg_8 = lookback_8[lookback_8['Sentiment'] == 'Negative']['Sentiment'].count()
            num_posts_all_10 = lookback_10['Sentiment'].count()

            info_list.append((ticker,
                              end.strftime('%Y-%m-%d'),
                              end.strftime('%H-%M-%S'),
                              num_posts_all,
                              num_posts_pos,
                              num_posts_neg,
                              num_posts_neg_8,
                              num_posts_all_10
                              ))
            del curr_ticker
            bar.next()

    current_table = pd.DataFrame(info_list, columns=['Ticker', 'Date', 'Time',
                                                     'Num_all_1',
                                                     'Num_pos_1',
                                                     'Num_neg_1',
                                                     'Num_neg_8',
                                                     'Num_all_10',])
    current_table['Rank_all'] = np.ceil(current_table['Num_all_1'].rank(axis=0, pct=True).mul(10)).astype(int)
    current_table['Rank_pos'] = np.ceil(current_table['Num_pos_1'].rank(axis=0, pct=True).mul(10)).astype(int)
    current_table['Rank_neg'] = np.ceil(current_table['Num_neg_1'].rank(axis=0, pct=True).mul(10)).astype(int)
    current_table['Rank_neg_8'] = np.ceil(current_table['Num_neg_8'].rank(axis=0, pct=True).mul(10)).astype(int)
    current_table['Rank_all_10'] = np.ceil(current_table['Num_all_10'].rank(axis=0,pct=True).mul(10)).astype(int)

    curr_Date = end.strftime('%Y-%m-%d')
    current_table.to_excel(f'//fileserver01/limdata/data/'
                           f'individual staff folders/andrew li/table_{_time}_{curr_Date}.xlsx', index=False)
    save_tosql(current_table, _time)

def save_tosql(df, which_table):
    from sqlalchemy import create_engine
    server = 'LIMHKDWH01S'
    user = 'andrew.li'
    password = 'an@lim355'
    DB = {'servername': server,
          'database': 'FORUM_DB',
          'driver': 'driver=SQL Server Native Client 11.0'}
    engine = create_engine(
        f'mssql+pyodbc://{user}:{password}@' + DB['servername'] + '/' + DB['database'] + "?" + DB['driver'])
    df['Ticker'] = df['Ticker'].astype('str')
    df.to_sql(f'table_{which_table}', engine, if_exists='append', index=False)


if __name__ == '__main__':
    # while True:
    #     try:
    #         if time.localtime().tm_hour == 13 and (time.localtime().tm_min == 0):
    #             update(15, num_cores=2)
    #             today = dt.datetime.now()
    #             prev_date = dt.datetime.now() - dt.timedelta(10)
    #             target_end_date = dt.datetime(today.year, today.month, today.day, 13)
    #             target_start_date = dt.datetime(prev_date.year, prev_date.month, prev_date.day, 15)
    #             create_current_summary_table(target_start_date, target_end_date, '1PM')
    #         elif (time.localtime().tm_hour == 14) and (time.localtime().tm_min == 30):
    #             update(-1, num_cores=4)  # If num_pages = -1, we will update the info page by page
                today = dt.datetime.now()
                prev_date = dt.datetime.now() - dt.timedelta(10)
                target_end_date = dt.datetime(today.year, today.month, today.day, 14, 30)
                target_start_date = dt.datetime(prev_date.year, prev_date.month, prev_date.day, 15)
                create_current_summary_table(target_start_date, target_end_date, '2-30PM')
    #         time.sleep(30)
    #
    #     except Exception as e:
    #         logging.exception('message')
    #         print(e)

