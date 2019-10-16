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
import re
import multiprocessing as mp
from progress.bar import Bar
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument('--log-level=3')
chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
thread_local = threading.local()
lock = threading.RLock()


# date = dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d")
date = '2019-10-15'

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
        self.time_req = 0
        self.total_pages = 1

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
        print(f'Parsing - {self._ticker}')
        soup = bs4.BeautifulSoup(page, 'html.parser')
        target_list = soup.find_all(class_='articleh normal_post')
        for target in target_list:
            try:
                links = target.find('span', class_='l3 a3').find_all('a')
                for link in links:
                    if re.match('/news,[\w\d]+,[\w\d]+.html', link.get('href')):
                        break
                text = target.text.strip('\n')
                readings, comments, title, author, published_time = text.split('\n')
                self.info_list.append((published_time, title, author, readings, comments, web_base+link.get('href')))
            except Exception as e:
                print(f'Parsing - {e}')

    def run(self):
        """
        For the stock, function will require all the pages and parse the website
        :return: True/False indicating that whether the stock is acquired successfully or not
        """
        global all_websites  # Declaim a global variable for downloading all the websites
        all_websites = {}
        first_page = f'http://guba.eastmoney.com/list,{self._ticker}_1.html'  # Use selenium to get the total pages
        try:
            with webdriver.Chrome('./chromedriver', options=chrome_options) as driver:
                driver.set_page_load_timeout(30)
                driver.get(first_page)
                soup = bs4.BeautifulSoup(driver.page_source, 'html.parser')
        except Exception as e:
            print(f'Error - {self._ticker} - {str(e)} - Stock.run')
            return False
        elements = soup.find_all('span', class_='sumpage')
        self.total_pages = int(elements[0].text)

        """
        Download all the websites and join them into a single string variable (all_in_one) for parsing
        """
        global bar
        bar = Bar(f'Scraping {self._ticker}', max=self.total_pages)
        sites = [f'http:/' \
                 f'/guba.eastmoney.com/list,{self._ticker}_{count}.html' for count in range(self.total_pages)]
        download_all_sites(sites)
        bar.finish()
        all_sites = sorted(all_websites.items(), key=lambda x: int(x[0]))
        all_in_one = ''.join([page for _, page in all_sites])
        _, time_parsing = self.parsing(all_in_one)
        print(f'Parsing Time - {int(time_parsing)} seconds')
        return True

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


def download_site(url):
    """
    Make sure the bar.next() will not be printed by different threads in the same time
    :param url: the url link to be scraped
    """
    lock.acquire()
    bar.next()
    lock.release()
    session = get_session()
    try:
        with session.request(method='GET', url=url, timeout=30) as response:
            try:
                count = re.findall('f_(\d+).html', url)[0]
            except Exception as e:
                count = url
            lock.acquire()
            all_websites[count] = response.text
            lock.release()
    except requests.exceptions.Timeout:
        print(f'\nTimeout - {url}')


def download_all_sites(sites):
    """
    Download all the pages to all_websites
    :param sites: the total sites to be scraped
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        executor.map(download_site, sites)


def reformat_date(df: pd.DataFrame):
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
    2018-09-09, 2017-09-07, 2016-10-06 will be re-formated to
    2018-09-09, 2018-09-07, 2016-10-06
    """
    df['cmp'] = np.where(df['diff'] <= dt.timedelta(0), True, False)
    indexs = df.index[df['cmp'] == False].values.tolist()
    links = []
    links_indexs = []
    for idx in indexs:
        if not re.match('.+qa_list.aspx', df.loc[idx, 'Link']):
            links.append(df.loc[idx, 'Link'])
            links_indexs.append(idx)
    # links = df['Link'][df['cmp'] == False].values.tolist()
    # links = [link for link in links if not re.match('.+qa_list.aspx', link)]  # exclude the strange link


    """
    Scrape the year info given the links
    """
    global all_websites
    global bar
    all_websites = {}
    bar = Bar('Formatting', max=len(links))
    target_years = []
    download_all_sites(links)
    bar.finish()

    try:
        """To handle the marginal case that two links are the same"""
        sorted_all_websites = [all_websites[link] for link in links]
    except Exception as e:
        print(f'Error - {str(e)} - links_reformatting')
        return

    for page in sorted_all_websites:
        try:
            soup = bs4.BeautifulSoup(page, 'html.parser')
            text = soup.find('div', class_='zwfbtime').text
            year = re.findall('([\d]+)-[\d]+-[\d]+', text)[0]
            target_years.append(year)
        except Exception as e:
            print(f'Error - {str(e)} - reformat_date - find_year_loop')
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
            print(f'Error - {str(e)} - reformat_date - reformat_loop')
            return
    return df



def run_by_date(ticker):
    """
    Scrape the stock info given its ticker
    :param ticker: the stock to be scraped
    :return: None
    """
    print('-' * 20, f'Doing {ticker}', '-' * 20)
    try:
        complete = False
        max_epoch = 5
        while (not complete) and (max_epoch > 0):
            max_epoch -= 1
            if not os.path.exists(f'data/historical/{date}/{ticker}.csv'):
                stock = Stock(ticker)
                complete = stock.run()
                if complete:
                    df = pd.DataFrame(stock.info_list, columns=['Time', 'Title', 'Author', 'Number of reads', 'Comments', 'Link'])
                    # df.to_csv(f'data/historical/{date}/{ticker}.csv', encoding='utf_8_sig', index=False)  # Can be removed
                    print(f'Finish Downloading {ticker}')
                    formated_df = reformat_date(df)
                    print(f'Finish Formatting {ticker}')
                    formated_df.to_csv(f'data/historical/{date}/{ticker}.csv', encoding='utf_8_sig', index=False)
    except Exception as e:
        print(f'Error - {ticker} - {str(e)} - run_by_date')


def run_by_multiprocesses():
    """
    Multiprocess function to speed up the program
    :return: None
    """

    os.chdir('C:/Users/andrew.li/Desktop/Projects/Lim/Project_2-Forum')
    if not os.path.exists(f'data/historical/{date}'):  # global variable: date
        os.mkdir(f'data/historical/{date}')
    # shanghai_list = pd.read_csv('data/target_list/SH.csv')
    # shanghai_list = shanghai_list.iloc[:, 0].apply(lambda x: str(x))
    # shanghai_list = shanghai_list.values.tolist()
    # shenzhen_list = pd.read_csv('data/target_list/SZ.csv')
    # shenzhen_list = shenzhen_list.iloc[:, 0].apply(lambda x: str(x).zfill(6))
    # shenzhen_list = shenzhen_list.values.tolist()

    csi300 = pd.read_excel('data/target_list/csi300_sz.xls', index_col=0).values[:, 3].tolist()
    csi300_list = []
    for ticker in csi300:
        if ticker >= 600000:
            csi300_list.append(f'{ticker}')
        else:
            csi300_list.append(f'{ticker}'.zfill(6))
    pool = mp.Pool(1)  # We may use multiple processes to speed up the program but progress bar will not appear properly
    pool.map(run_by_date, csi300_list)
    # with Bar('Processing', max=len(shanghai_list + shenzhen_list)) as bar:
    #     for _ in pool.imap_unordered(run_by_date, shanghai_list + shenzhen_list):
    #         bar.next()


if __name__ == '__main__':
    run_by_multiprocesses()
    # df = pd.read_csv(f'data/historical/{date}/')
    # reformat_date()
