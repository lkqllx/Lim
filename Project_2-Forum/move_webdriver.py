import os
from shutil import copy

if __name__ == '__main__':
    if not os.path.exists('C:/webdriver/'):
        os.makedirs('C:/webdriver/')
    copy('./chromedriver.exe', 'C:/webdriver/')