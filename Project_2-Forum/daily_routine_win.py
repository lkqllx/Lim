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
from openpyxl.styles import colors
from openpyxl.formatting.rule import DataBarRule

logging.basicConfig(filename=f'logs/daily_routine_{dt.datetime.now().year}{dt.datetime.now().month}'
                             f'{dt.datetime.now().day}_{dt.datetime.now().hour}{dt.datetime.now().minute}.log',
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

def get_proxy():
    proxies = []
    url = 'https://free-proxy-list.net/'
    web = requests.get(url)
    soup = bs4.BeautifulSoup(web.content, 'lxml')
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

    def __init__(self, ticker):
        self._ticker = ticker
        self.info_list = []
        self.time_req = 0
        self.total_pages = 1
        self.all_websites = {}
        self.thread_local = threading.local()

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
                readings = all_info[0]
                comments = all_info[1]
                author = all_info[-2]
                published_time = all_info[-1]
                title = ''.join(all_info[2:-2])
                self.info_list.append((published_time, title, author, readings, comments, web_base+link.get('href')))
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
                # soup = bs4.BeautifulSoup(driver.page_source, 'html.parser')
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
        time_web = int(time.time() - start)
        """
        Download all the websites and join them into a single string variable (all_in_one) for parsing
        """
        time_parsing = time.time()
        sites = [f'http:/' \
                 f'/guba.eastmoney.com/list,{self._ticker},f_{count}.html' for count in range(10) if count != 1]
        self.download_all_sites(sites)
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
            with session.request(method='GET', url=url, timeout=30) as response:
                try:
                    count = re.findall('list,[\w\d]+,f_(\d+).html', url)[0]
                except Exception as e:
                    count = url
                lock.acquire()
                self.all_websites[count] = response.text
                lock.release()
        except requests.exceptions.Timeout:
            print(f'Timeout - {url}')
            logging.exception(f'Timeout - {url}')

    def download_all_sites(self, sites):
        """
        Download all the pages to all_websites
        :param sites: the total sites to be scraped
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(self.download_site, sites)

    def reformat_date(self, df: pd.DataFrame):
        """
        Since there is no year information in the dataset, we need to add this info
        Currently implementation is to compare the two dates. If the later one is greater than
        the previous one, the year will be reduce one.
        :param df: pd.DataFrame which contains the scraped info
        :return: pd.DataFrame which add year info to the time
        """

        df['Time_to_cmp'] = df['Time'].apply(lambda x: '{}-'.format(2000) + x)
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
                    df.loc[links_indexs[idx]:, 'Time'] = \
                        df.loc[links_indexs[idx]:, 'Time'].apply(lambda x: '{}-'.format(target_years[idx]) + x)
                else:
                    """Not last index, we will add year info between links_indexs[idx]:links_indexs[idx+1]"""
                    df.loc[links_indexs[idx]:links_indexs[idx+1]-1, 'Time'] =  \
                        df.loc[links_indexs[idx]:links_indexs[idx+1]-1, 'Time'].apply(lambda x: '{}-'.format(target_years[idx]) + x)
            except Exception as e:
                logging.error(f'Error - {str(e)} - reformat_date - reformat_loop', exc_info=True)
                return
        return df


def run_by_date(args):
    """
    Scrape the stock info given its ticker
    :param ticker: the stock to be scraped
    :return: None
    """
    # print('-' * 20, f'Doing {ticker}', '-' * 20)
    ticker, dates = args
    complete = False
    max_epoch = 5
    while (not complete) and (max_epoch >= 0):
        try:
            max_epoch -= 1
            stock = Stock(ticker)
            complete, time_parsing, time_web = stock.run()
            if complete:
                df = pd.DataFrame(stock.info_list, columns=['Time', 'Title', 'Author',
                                                            'Number of reads', 'Comments', 'Link'])
                # print(f'Finish Downloading {ticker}')
                formated_df = stock.reformat_date(df)
                formated_df['Time'] = formated_df['Time'].apply(lambda row: dt.datetime.strptime(row, '%Y-%m-%d %H:%M'))
                filtered_df = formated_df[(formated_df['Time'] >= dates[0]) &
                                          (formated_df['Time'] < dates[1])]
                current_number_posts = filtered_df['Time'].count()
                if current_number_posts == 0:
                    print(f'{ticker}')
                return (ticker, current_number_posts, time_web, time_parsing)
            else:
                logging.info(f'Missing the Value of {ticker}')

        except Exception as e:
            logging.error(f'Error - {ticker} - {str(e)} - run_by_date', exc_info=True)
            complete = False
            continue
    return (ticker, -1, 0, 0)



@timer
def run_by_multiprocesses(csi300, dates):
    """
    Multiprocess function to speed up the program
    :return: None
    """
    results = []
    time_parsing = 0
    time_web = 0
    with Bar('Downloading', max=len(csi300)) as bar:
        with mp.Pool(8) as pool:
            for output in pool.imap_unordered(run_by_date, zip(csi300, [dates] * len(csi300))):
                bar.next()
                results.append(output[:2])
                time_web += output[2]
                time_parsing += output[3]
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


if __name__ == '__main__':
    # while True:

    today = dt.datetime.now()
    yesturday = dt.datetime.now() - dt.timedelta(1)
    if not os.path.exists('data/current_list.pkl'):
        jq.auth('18810906018', '906018')
        csi300 = jq.get_index_stocks('000300.XSHG', dt.datetime.strftime(dt.datetime.now(), format='%Y-%m-%d'))
        with open('data/current_list.pkl', 'wb') as f:
            pickle.dump(csi300, f)
    else:
        with open('data/current_list.pkl', 'rb') as f:
            csi300 = pickle.load(f)
    csi300 = [ticker.split('.')[0] for ticker in csi300][:300]
    # csi300 = ['000069']
    # if time.localtime().tm_hour == 14 and time.localtime().tm_min == 30:
    if True:
        try:

            target_end_date = dt.datetime(today.year, today.month, today.day, 14, 30)
            target_start_date = dt.datetime(yesturday.year, yesturday.month, yesturday.day, 15)
            dates = list([target_start_date, target_end_date])
            (curr_list, time_web, time_parsing), time_used = run_by_multiprocesses(csi300, dates)
            print(f'Time Elapsed - {time_used}')
            print(f'Time Web - {time_web}')
            print(f'Time Parse - {time_parsing}')
            all_succesful = all(np.not_equal(list(zip(*curr_list))[1], -1))
            succesful_list = [(ticker, post) for ticker, post in curr_list if post != -1]
            not_succesful_list = [ticker for ticker, post in curr_list if post == -1]
            new_curr_list = []
            count = 5
            while (not all_succesful) and count >= 0:
                print('Failed Ticker', f'\n, '.join(not_succesful_list))
                (new_curr_list, time_web, time_parsing), time_used = run_by_multiprocesses(not_succesful_list, dates)
                all_succesful = all(np.not_equal(list(zip(*new_curr_list))[1], -1))
                count -= 1
            succesful_list.append(new_curr_list)
            write_into_excel(curr_list)

        except Exception as e:
            logging.exception('message')
            print(e)

