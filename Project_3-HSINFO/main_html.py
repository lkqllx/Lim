"""
This repository is designed for scraping information of
https://www.hsi.com.hk/eng/newsroom/index-other-notices
which is the website for announcing Hang Seng index changes
"""
import bs4
import requests
import pandas as pd
import datetime as dt
import os
from selenium import webdriver
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import textwrap
import hashlib
import sys
import time
from selenium.webdriver.chrome.options import Options
sys.stdout.flush()
chrome_options = Options()
chrome_options.add_argument("--headless")


EMAIL_RECEIVERS = ['simon.chan@limadvisors.com', 'andrew.li@limadvisors.com']
ALTERNATE_EMAIL_RECEIVER = 'andrew.li@limadvisors.com'
EMAIL_SENDER = 'andrew730.li@gmail.com'
EMAIL_SENDER_PASSWORD = 'LlX13985112851'


class Crawler:
    def __init__(self, web):
        self._web = web

    def get_soup(self):
        page = requests.get(self._web)
        self.soup = bs4.BeautifulSoup(page.text, 'html.parser')

    def find_pattern(self, pattern, pattern_cls=None):
        return self.soup.find_all(pattern, pattern_cls) if pattern_cls else self.soup.find_all(pattern)

    def find_cls(self, cls):
        return self.soup.find_all(class_=cls)


def send_email_by_smtp(subject="", body=""):
    smtp_server = "smtp.gmail.com"
    port = 465  # For starttls
    sender_email = EMAIL_SENDER
    password = EMAIL_SENDER_PASSWORD

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        # server = smtplib.SMTP(smtp_server, port)
        # server.ehlo() # Can be omitted
        # server.starttls(context=context)  # Secure the connection
        # server.ehlo() # Can be omitted
        # server.login(sender_email, password)
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            if body is None:
                receiver_email = [ALTERNATE_EMAIL_RECEIVER,]
                html = "<p>No update</p>"
            else:
                receiver_email = EMAIL_RECEIVERS
                # Create the body of the message (HTML version).
                html = body
            # Create message container - the correct MIME type is multipart/alternative.
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = sender_email
            for receiver in receiver_email:
                msg['To'] = receiver

            # Record the MIME types of both parts - text/plain and text/html.
            part1 = MIMEText(html, 'html')

            # Attach parts into message container.
            # According to RFC 2046, the last part of a multipart message, in this case
            # the HTML message, is best and preferred.
            msg.attach(part1)

            for receiver in receiver_email:
                server.sendmail(sender_email, receiver.split(','), msg.as_string())
    except Exception as e:
        # Print any error messages to stdout
        print(str(e))



def create_HTML_title(title):
    return '''<html>
                <head>
                    <h1> {} </h1>
                </head>
              </html>
            '''.format(title)


def check_update(new_record):
    """
    Check whether there is an update in new_record compared with our own record

    Another function of check_update is create a uniform id for each item
    by using hash function:
                    unique id = hash(Date + Title + Link)

    :param new_record: latest DataFrame scraped from website
    :return: a list which contains the difference to send email
    """
    diffs = []
    if not os.path.exists('data/old_record.csv'):
        # Initial case
        # TODO: Something wrong about the string format for hashing

        new_record['Id'] = new_record.apply(lambda x: hashlib.md5((x[0]+x[1]+x[2]).encode('utf-8')).hexdigest()[:10], axis=1)
        new_record.to_csv('data/old_record.csv', index=False)
    else:
        old_record = set(pd.read_csv('data/old_record.csv')['Id'].values.tolist())
        for idx, text in enumerate(new_record.values):
            if hashlib.md5((text[0]+text[1]+text[2]).encode('utf-8')).hexdigest()[:10] not in old_record:
                diffs.append(new_record.values[idx, :])

        # Update record no matter what
        new_record['Id'] = new_record.apply(lambda x: hashlib.md5((x[0]+x[1]+x[2]).encode('utf-8')).hexdigest()[:10], axis=1)
        new_record.to_csv('data/old_record.csv', index=False)
        return diffs


def send_email(diffs):
    """
    Create html for sending the email and wrap the long text
    :param diffs: the difference between new and old record
    :return: None
    """
    if diffs:
        for date, msg, link in diffs:
            wrapped_lines = textwrap.wrap(msg)
            new_title = '<br>'.join(wrapped_lines)
            html = f"""
            <html>
              <body>
                <p>Hi Simon,<br>
                    <p>There is an update in Hang Seng Index on {date}</p>
                    <p>It contains following message:</p>
                    <b>{new_title}</b>
                    <p>You may click <a href="{link}">here</a> to view the documentation.</p>
                    <p>Best Regards,</p>
                    <p>Andrew</p>
                </p>
              </body>
            </html>
            """
            title = 'An Update of Hang Seng Index Constituents'
            print(f'There is an update and email was sent at '
                  f'{dt.datetime.now().hour}:{dt.datetime.now().minute}:{dt.datetime.now().second}, '
                  f'{dt.datetime.now().year}-{dt.datetime.now().month}-{dt.datetime.now().day}')
            send_email_by_smtp(title, body=html)
    else:
        print(f'There is no update on {dt.datetime.now().hour}:{dt.datetime.now().minute}:{dt.datetime.now().second}, '
              f'{dt.datetime.now().year}-{dt.datetime.now().month}-{dt.datetime.now().day}')


def run_once():
    web = 'https://www.hsi.com.hk/eng/newsroom/index-other-notices'
    with webdriver.Chrome('C:/webdriver/chromedriver.exe', options=chrome_options) as driver:
        driver.set_page_load_timeout(120)
        driver.get(web)
        soup = bs4.BeautifulSoup(driver.page_source, 'html.parser')
    elements = soup.find_all('div', class_='newsItem clearFix')
    prefix_html = 'https://www.hsi.com.hk'
    outs = []
    for element in elements:
        text = element.text.strip('\n')
        text_list = text.split('\n')
        date = text_list[0]
        title = text_list[1]
        link = element.find(class_='btnViewNow').get('href')
        outs.append([date, title, prefix_html+link])
    new_record = pd.DataFrame(outs, columns=['Date', 'Title', 'Link'])
    diffs = check_update(new_record)
    send_email(diffs)


def run():
    print('Running......')
    """Check for every default time"""
    while True:
        try:
            run_once()
        except Exception as e:
            print(f'Unexpected error - {str(e)}')
        time.sleep(60 * 10)  # 60 secs (1 minute) * 10 -> 10 mins


if __name__ == '__main__':
    run()
