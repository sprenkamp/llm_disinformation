# copied and slighty modified from https://github.com/ChangyWen/PolitiFact-scraping/blob/master/scrape-politifact.py
#TODO possibly contact https://arxiv.org/pdf/1705.00648.pdf%E2%80%8B if data should be updated

#Import the dependencies
from bs4 import BeautifulSoup
import pandas as pd
import requests
import urllib.request
import time
from tqdm import tqdm

#Create lists to store the scraped data
authors = []
dates = []
statements = []
sources = []
targets = []

#Create a function to scrape the site
def scrape_website(page_number, source=None):
    page_num = str(page_number) #Convert the page number to a string

    '''source: all'''
    URL = 'https://www.politifact.com/factchecks/list/?page='+page_num #append the page number to complete the URL
    if source is not None:
        '''source: a certain speaker only'''
        URL = 'https://www.politifact.com/factchecks/list/?page={}&speaker={}'.format(page_num, source)
    
    webpage = requests.get(URL)  #Make a request to the website
    #time.sleep(3)
    soup = BeautifulSoup(webpage.text, "html.parser") #Parse the text from the website
    #Get the tags and it's class
    statement_footer =  soup.find_all('footer',attrs={'class':'m-statement__footer'})  #Get the tag and it's class
    statement_quote = soup.find_all('div', attrs={'class':'m-statement__quote'}) #Get the tag and it's class
    statement_meta = soup.find_all('div', attrs={'class':'m-statement__meta'})#Get the tag and it's class
    target = soup.find_all('div', attrs={'class':'m-statement__meter'}) #Get the tag and it's class
    # source_name = soup.find_all('div', attrs={'class':'m-statement__meter'}) #source where the statement was published
    #loop through the footer class m-statement__footer to get the date and author
    for i in range(len(statement_footer)):
        try:
            # Extracting author's name
            link1 = statement_footer[i].text.strip()
            name_and_date = link1.split()
            first_name = name_and_date[1]
            last_name = name_and_date[2]
            full_name = first_name + ' ' + last_name

            # Extracting date
            month = name_and_date[4]
            day = name_and_date[5]
            year = name_and_date[6]
            date = month + ' ' + day + ' ' + year

            # Extracting statement
            link2 = statement_quote[i].find_all('a')
            statement_text = link2[0].text.strip()

            # Extracting source
            link3 = statement_meta[i].find_all('a')
            source_text = link3[0].text.strip()

            # Extracting fact
            fact = target[i].find('div', attrs={'class': 'c-image'}).find('img').get('alt')

            # Append the data to the respective lists
            dates.append(date)
            authors.append(full_name)
            statements.append(statement_text)
            sources.append(source_text)
            targets.append(fact)

        except IndexError:
            # Skip this iteration if any error occurs
            continue

#Loop through 'n-1' webpages to scrape the data
n=797 #NOTE 797 is the last page as of 24.11.23
for i in tqdm(range(1, n)):
    scrape_website(i, source='joe-biden')

#Create a new dataFrame
data = pd.DataFrame(columns = ['author',  #person who fact-checked the statement
                                'statement', #the statement
                                'source', #source where the statement was published this can be a person or an organization
                                'date', #date when the statement was published
                                'target'] #label i.e. Pants on Fire, False, Mostly False, Half-True, Mostly True, True
                                )
data['author'] = authors 
data['statement'] = statements
data['source'] = sources
data['date'] = dates
data['target'] = targets
#Show the data set
# print(data)
data.to_csv('data/politifact/sample.csv', index=False, sep=',')