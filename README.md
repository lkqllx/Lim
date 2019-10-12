# Hang Seng Index Newsroom Crawler
This repository is created by *Andrew LI* for scraping the information at https://www.hsi.com.hk/eng/newsroom/index-other-notices.
The information of the website will be scraped into a csv file named "old_record", which will
be compared with the latest one. 

If *Yes* -> send the email to Simon

If *No* -> sleep for 10 minutes and re-run the whole process

## Execution
There different ways to execute the project:

`python main_html.py`: You may run the program in the trivial way but make that 
all the dependencies are well installed by `pip install -r requirements.txt`.

`nohup python -u main_html.py & > nohup.out &`: The program will be run silently and
outputs will be overwritten into *nohup.out*.


`docker build --tag hsinfo .` + `docker run hsinfo`
It will work if you have installed Chrome in your _IMAGE_.


*Please note that the Chrome must be installed in your operating system*



