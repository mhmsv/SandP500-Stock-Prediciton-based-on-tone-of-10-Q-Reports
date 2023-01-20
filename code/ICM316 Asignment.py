
import requests
import pandas as pd
#defining a list to iterate over quarters
quarter = ['QTR1','QTR2','QTR3','QTR4']

#Header with my laptop User Agent
heads = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15',
'Accept-Encoding':'*',
'Host':'www.sec.gov'
         }

# Question 1 
# Download all fillings in 2021 from the U.S Securities and Exchange Commission Electronic Data Gathering, Analysis and Retrieval system (EDGAR).
for q in quarter: #iterate over SEC EDEGAR to download master index quarterly
    url = r"https://www.sec.gov/Archives/edgar/full-index/2021/"+q+"/master.idx"
    print(url)
    
    request_content = requests.get (url, headers = heads).content
    print(type (request_content))
    #opening a new file to write downloaded data into disk
    result = request_content.decode ("utf-8", "ignore")
    with open('master_index_2021_'+q+'.txt', 'w') as f:
        f.write(result)

#opening master index file
data_2021 = [] #storing whole data, all quarters together

for q in quarter:
    #opening master index files quarterly
    with open('master_index_2021_'+q+'.txt', 'r') as f:
        linesoffile=f.readlines()
    records = [tuple(line.split('|')) for line in linesoffile [11:]]
    for r in records:
        data_2021.append([r[0], r[1], r[2], r'https://www.sec.gov/Archives/'+r[4]])
#building one dataframe consisting all  of our quarterly data with respected columns      
df_2021 = pd.DataFrame(data_2021 , columns = ['cik','firm','date' , 'url'])
print(df_2021)



#print our data frame and its shape rows and cols
print(df_2021.shape)
print(df_2021.head()) #check that data is stored perfectly or not

#Question 2 
#(2) Create a Structural Query Language (SQL) database to store all fillings in 2021 from EDGAR. Discuss the database e.g. how many firms? How many fillings per firm on average?  

import sqlite3 #importing sqllite for sql operation

connection = sqlite3.connect('Edgar2021_database.db')  #creating a connection and creating  database to store filling into in
c = connection.cursor() 
                   

c.execute('CREATE TABLE IF NOT EXISTS Edgar_table (cik number, firm text, date text, url text)') #Query to create Edegar_table with columns same as columns of our dataframe
connection.commit() #commiting our query into swl

#transfoming our dataframe into sql format using to_sql() functuion, given values are name of table that we want to push our df into in
# and our active connection to sql
df_2021.to_sql('Edgar_table', connection, if_exists='replace', index = False)

#quering Select * on Edegar_Table to check that values of df are stored successfully or not
c.execute('''  
SELECT * FROM Edgar_table
          ''')
#printing result of our query
for row in c.fetchall():
    print (row)

# how many firms
c.execute('''  
SELECT COUNT(DISTINCT firm) FROM Edgar_table
          ''')
#printing result of our query
for row in c.fetchall():
    print("different  firm")
    report_count = row[0]
    print (row[0])

# how many reports in date colmn in total
c.execute('''  
SELECT COUNT(date) FROM Edgar_table
          ''')
#printing result of our query
for row in c.fetchall():
    print("Number of REPORTS")
    report_count = row[0]
    print (row[0])

print("average report that firms have: ", report_count/firm_count)

#question 3 
#(3) Query 10-Q filings from the database.
#query only #date = 10-Q to
sql_query = pd.read_sql_query ('''
                                    SELECT * FROM Edgar_table
                                    WHERE date = '10-Q'
                               ''', connection)

#creating a datafrom from query and store date = 10q on
df_10Q = pd.DataFrame(sql_query, columns = ['cik', 'firm', 'date','url'])
print (df_10Q)

#printing our df to check eerythin gis fine
print (df_10Q)

#Question4
#loading cmpant tickers, in order to match cik to ticker using https://www.sec.gov/file/company-tickers





#Question4
#untill now we have only cik and firm names of companies, in order to download their price data we need their respective tickers
#to find what ticker is for which company, we download company tikers from  https://www.sec.gov/file/company-tickers
#then we try to match ticker based on cik
import json
import pandas as pd
ticker_data = pd.DataFrame(columns=['cik','Ticker']) #creating data frame to store ticker and cik form company-tickers.json file
with open('company_tickers.json') as data_file:    #opening downloaded file
    data = json.load(data_file)
    for v in data.values():
        data2 = {
                  "cik": v['cik_str'],
                  "Ticker": v['ticker']
                }
        ticker_data = ticker_data.append(data2,ignore_index=True) #appendig all of lines

print(ticker_data) # printing company tickers and cik numbers



#now we are going to merge ticker into df_10q (question3 df)
#convert cik of ticker df to int type in order to join the df on column 'cik'  with df_10q df
ticker_data['cik'] = ticker_data['cik'].apply(str)

#creating a datafrom from query same as q3 
#reading it from our local db
import sqlite3

connection = sqlite3.connect('Edgar2021_database.db') 
c = connection.cursor()
sql_query = pd.read_sql_query ('''
                                    SELECT * FROM Edgar_table
                                    WHERE date = '10-Q'
                               ''', connection)

df_10Q = pd.DataFrame(sql_query, columns = ['cik', 'firm', 'date','url']) #creating df_10q same as previous part
print (df_10Q)

#print(type(df_10Q.iloc[0,0]),(df_10Q.iloc[0,0]))
#print(type(ticker_data.iloc[0,0]))

#iterate over cik to find ticker names
import numpy as np
df_t = pd.DataFrame()
count = 0 # to have line number
#print(ticker_data['Ticker'].loc[ticker_data['cik'] == 1001115])
#print(type(str(ticker_data['Ticker'].loc[ticker_data['cik'] == 1001115])))
for i in df_10Q['cik']:
    count = count + 1

#join on column
#joiing question3(df_10q) df and ticker df together based on cik 
df_final_ticker_cik=pd.merge(df_10Q, ticker_data, on='cik', how='inner')

print(df_final_ticker_cik)



#now we have ticker of all 10-Q firms in df_final_ticker_cik_notNA
#droping NA  there were only 11737 in company ticker of sec gov, but we had 25567 rows in  10-q df, however some of them were duplicate, at the end we found 22739 ticker out of 25567

df_final_ticker_cik_notNA = df_final_ticker_cik[df_final_ticker_cik['Ticker'].notna()]

df_final_ticker_cik_notNA

#find url of sample firm to test
print(df_final_ticker_cik_notNA.iloc[1,3])

# select 50 firms out of whole list to download historical price data, we can change n and download as much as we want 
#, and code would still work, miniumn required number was 30, we choose 50 
#the order of cods will increase as we increase n and it needs more time to download data and also process further questuons codes on it
df_ticker_price_sample = df_final_ticker_cik_notNA.sample(n=50)

print(df_ticker_price_sample) #printing our sample

print(df_ticker_price_sample['Ticker']) #printing sample tickers

#making df of sample firms
#creating this for easier access through samples ticker name in order to iterate into it to download their price
df_sample_ticker = pd.DataFrame(df_ticker_price_sample['Ticker'] , columns = ['Ticker'])

print(df_sample_ticker) # our sample tickers



print(df_final_ticker_cik_notNA) #df of all firm

df_final_ticker_cik_notNA.rename(columns={'ticker':'Ticker'}, inplace=True) #rename column name in order to join that with our sample tickers based on ticker coulmn

#buliding our final df of firms with 10q reports in aditon to their tickers and alsso crearting new variable to show date of report
df_final_sample_10q =pd.merge(df_final_ticker_cik_notNA, df_sample_ticker, on='Ticker', how='right')

#giving intial value for date of report
df_final_sample_10q['date of report'] = 0

print(df_final_sample_10q) #priting our final df with 10q reports in aditon to their tickers and alsso date of report

#question 4 (4) Download stock prices of some companies that filed 10-Q forms in 2021. Note: You are NOT required to download stock prices 
#of all the firms that filed 10-Q forms in 2021. The minimum number of firms is 30.  There is merit to explain your choice of the sampled firms.
#Downloading historical prices of all tickers in sample from yahoo finance by iterating over ticekr colmn
#!pip install yfinance

from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override()

df_ticker_price = pd.DataFrame()
for ticker in df_ticker_price_sample['Ticker']:
    df = pdr.get_data_yahoo(ticker, start="2021-01-01", end="2021-12-31") #period of year 2021
    df['ticker'] = ticker
    df_ticker_price = df_ticker_price.append(df) #appending historical price of our sample firms line by line

# adding risk market and risk free data to our data
#s&p 500 is ^GSPC and IRX as risk free rate, trasury bill index
#store them in temporary df called df_tmp
df_tmp = pdr.get_data_yahoo('^GSPC', start="2021-01-01", end="2021-12-31")
df_tmp['ticker'] = "^GSPC"
df_ticker_price = df_ticker_price.append(df_tmp) #appending tmp df to our price df

df_tmp = pdr.get_data_yahoo('^IRX', start="2021-01-01", end="2021-12-31")
df_tmp['ticker'] = "^IRX"
df_ticker_price = df_ticker_price.append(df_tmp)#appending tmp df to our price df

print(df_tmp)

#CREATE A RETURN USING LOG FUNCTION AND SHIFTNG A CELL UP, doing that in a temprary df called df_copy
df_copy = df_ticker_price
df_copy['Return'] = np.log(df_copy['Adj Close']/df_copy['Adj Close'].shift(1))
print(df_copy)



#pivoting our temporary data(df_copy) to making it right name of ticker on top and their regarded return on fornt as a value, now we will have ticker names as column and also their return on each day in addition to risk free and market portfolio
pivot_data = df_copy.pivot(columns='ticker',values='Return')
pivot_data.head(10)
print(pivot_data)

#RENAMINING NAME OF MARKET portfolio ticker AND RISK FREE RATE
pivot_data.rename(columns={'^GSPC':'Rm'}, inplace=True)
pivot_data.rename(columns={'^IRX':'Rf_Rate'}, inplace=True)



rm_rf_ticker_return_df = pivot_data #assigining our ivoted df into our new df with every price inside of it

rm_rf_ticker_return_df['Rm_Rf']= rm_rf_ticker_return_df['Rm']-rm_rf_ticker_return_df['Rf_Rate'] #creating rm-rf column with substracting rm rf values
rm_rf_ticker_return_df['Rf_Rate'] = rm_rf_ticker_return_df['Rf_Rate']/250 #adjusting rf by dividing by 250 same as seminars

print(rm_rf_ticker_return_df) #printing our full price df with everything



df_ticker_price=df_ticker_price.reset_index() # to reset index and move date time to other column

df_ticker_price['date of report'] = 0
#adding data of report column to put 'Date' column value into it but as a string ! this date of report is same as date of price not 10-q report!

df_ticker_price



#do this for when we want to merge df ticker price, with df10q sample on date column, both date should be same type , string !
count = 0
#just copying Date cloumn into date of report column as string "yyyy-mm-dd" this is date of reported price not 10-q report date !
for date in df_ticker_price['Date']:
    date = str(date)[:10]
    df_ticker_price.iloc[count,9] = date
    count +=1

df_ticker_price

#question 5 merging 
#renaming Ticker into ticker, in order to join as a key
df_final_ticker_cik_notNA.rename(columns={'Ticker':'ticker'}, inplace=True)
df_ticker_price.rename(columns={'Ticker':'ticker'}, inplace=True)

df_final_ticker_cik_notNA.rename(columns={'Ticker':'ticker'}, inplace=True)

#joining our price df and previous df (q3) together to have all of sample firms with their price on the required period
df_price_merge_ticker_cik=pd.merge(df_ticker_price, df_final_ticker_cik_notNA, on='ticker', how='inner')

print(df_price_merge_ticker_cik)
#merged with price



# use df_final_sample_10q to download 10-q

import requests
import os
#now iterating through url columns of our sample firms to download qoq reports and store them on drive
heads = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15',
'Accept-Encoding':'*',
'Host':'www.sec.gov'
         }

count =0; #to find name of firm in a same row as url

for url in df_final_sample_10q['url']:
    url = url[:-1]     #preventing /n
    request_content = requests.get (url, headers = heads). content # requesting based on url of our sample to download filling
    result = request_content.decode ("utf-8", "ignore")
    #having a specifi formating for reports using .format
    file_name = '{0}__{1}_(url:{2}).txt'.format(df_final_sample_10q.iloc[count,4],df_final_sample_10q.iloc[count,2],count).replace('/','|') #should replace / with | in order to prevent issues in file pathing
    #writing files into disk with coresponded name
    with open (os.path.join('/content/downloaded_reports/',file_name),'w') as f:
        f.write(result)
    print(file_name)
    count +=1;

# now to find exact date of report
#up untill now we only have date of reported price, now we need to find on whihc date reports were release
#dive into every reports to find date of each of them

#if we open filling in disk we will e dates ,year month and day of reportet report were given into line 7-8
#now we will open every file and read that value and write it in date of report column (this is date of 10-q report filling !)!!!!
import datetime
import time
from datetime import datetime

count =0; #to find name of firm in a same row as url
# iterate into files to find date of report from 10-q reports
for url in df_final_sample_10q['url']:
    url = url[:-1]     #preventing /n
    file_name = '{0}__{1}_(url:{2}).txt'.format(df_final_sample_10q.iloc[count,4],df_final_sample_10q.iloc[count,2],count).replace('/','|') #agar ino nazarim fekr mikone folder bode va adrese path boode
    with open (os.path.join('/content/downloaded_reports/',file_name),'r') as f:
            linesoffile=f.readlines()
            date =  str((linesoffile [7:8]))
            year = date[23:27]
            month = date[27:29]
            day = date[29:31]
            #conver dt to string for when we want to mergedf_fnal sample with df_ticket price
            date = year+"-"+month+"-"+day
            df_final_sample_10q.iloc[count,5] = date
            #print(type(date))
    
    count +=1;



# reading dictionary mcdonald from disk
lexicon = pd.read_csv("/content/Loughran-McDonald_MasterDictionary_1993-2021_simplified_ver.csv")
print(lexicon.head(10))

# masking for negative words:
mask = lexicon['negative'] == 1
negatives =  lexicon[mask]['Word']
print(negatives)

# making mask for postive
mask = lexicon['positive'] == 1
positives = lexicon[mask]['Word']
print(positives)

#change them to list
pos_words = positives.tolist()
neg_words = negatives.tolist()

#creating new column for sentiment of each report and initiating it with zero # ####inja bodam
df_final_sample_10q['sentiment_of_report'] = 0

print(df_final_sample_10q)

print(df_final_sample_10q.iloc[0,5])

#function to clean report
# Cleaning the report using nltk library
import re 
import nltk
nltk.download('stopwords') #downloadin stopwords
nltk.download('words') 
ps = nltk.stem.porter.PorterStemmer()

stopwords = set(nltk.corpus.stopwords.words('english')) #loading stop words

def clean(report):
    report = report.lower()
    report = re.sub("[^a-z]", " ", report) #keep only a-z and spaces in each report
    words = report.split()
    words = [word for word in words if word not in stopwords] #remove stopwords in each report
    words = [ps.stem(word) for word in words] # lemmatize/stem words
    report = ' '.join(words)
    return report

"""# New Section"""

#function to get sentiment of report by counting positive and negative words in eaach of them
def get_sentiment(words):
    try:
        neg_count = 0
        pos_count = 0
        for word in words:
            word = word.upper() # to upper case
            if (word in neg_words): 
                neg_count+=1
            if (word in pos_words):
                pos_count += 1
        if (pos_count == 0 and neg_count == 0):
            print(pos_count,neg_count )
            return("No Negative and Positive Word")
        print("pos count  : " ,pos_count, " neg count : " ,neg_count )
        return((pos_count - neg_count) / (pos_count+neg_count))
    except:
        print("An exception occurred")  #in case exception occured, return this for our record and log and also prevent crash of whole code
        return  ("An exception occurred" )

#question 6
#going through whole downloaded file on disk 
#1- create file name and its path
#2- parse its html format ot text using beautiful soup library
#3- cleaning words
#4-applying get sentiment function to check negative postive words
#5-assigning dummy for negative or postive dummy in sentimnet column
# in the ouyput we will see name of the file and also count of pos/neg and total ratio of it
from bs4 import BeautifulSoup
count = -1 
for url in df_final_sample_10q['url']:
    count+=1
    
    try:   
        url = url[:-1]     #preventing /n
        file_name = '{0}__{1}_(url:{2}).txt'.format(df_final_sample_10q.iloc[count,4],df_final_sample_10q.iloc[count,2],count).replace('/','|') #agar ino nazarim fekr mikone folder bode va adrese path boode
        print(file_name)
        with open (os.path.join('/content/downloaded_reports/',file_name),'r') as f:
            file = f.read()
            words = file.split()
        soup = BeautifulSoup(file,'html.parser')
        content = soup.get_text()
        words = file.split()
        #words = clean(words)
        sentiment_number = get_sentiment(words)
        print(sentiment_number)
        if(sentiment_number>=0): #seting positive dummy
            df_final_sample_10q.iloc[count,6] = 1
        else:
            df_final_sample_10q.iloc[count,6] = -1 #setting negative dummy

    except:
        df_final_sample_10q.iloc[count,6] = "error"
        continue

df_final_sample_10q

#check that calye of all reports are given
dummy_df = df_final_sample_10q[['Ticker', 'date of report','sentiment_of_report']].copy()
print(dummy_df)



#rename to join on cloumn
df_ticker_price.rename(columns={'ticker':'Ticker'}, inplace=True)

#(7) Produce a summary statistics of these companies’ stocks characteristics, e.g., 
#returns, trading volume, liquidity, etc. during positive/negative filing dates.

#question 7 apply summary statics

#ceating df to merge price of stock on dates that we have dummy = 1 or -1 for question 7 to apply statistics
#using 2value as key to have only dates with dummy
df_price_dummy_q7 =pd.merge(dummy_df, df_ticker_price, on=['date of report','Ticker'], how='left')

print(df_price_dummy_q7)

# in order to do summary statstics for each firm specifically on postive/negative period (for most of them there is 3 period)
# we put one condition to filter whole price data, only for one specif firm, then load price history of all dates with report(have dummy) then apply describe function to find
# in output we can see summary statstics for all companies n our sample(n=50 companies ) data only on postive/negative period

for i  in df_ticker_price_sample['Ticker']: #iterating over sample firms
    condition = df_price_dummy_q7['Ticker'] == i
    df_price_dummy_q7_tmp  = df_price_dummy_q7[condition]
    
    print("summary statistics during negative/postive filling dates for firm " , i, " : \n\n")
    print(df_price_dummy_q7_tmp.describe(),"\n\n")

#q8
#(8) Conduct an econometric procedure to examine the relationship between any of the stock characteristics and tone in 10-Q filings.
#merging dummy df with our price df to have verything in addition to dummies
#merging to apply regression having data set of all dummies with -1 or 0 or 1
df_tickerPrice_sentimentDummy =pd.merge(df_ticker_price, dummy_df, on=['date of report','Ticker'], how='left')

#assigingin zero to rows without dummy
df_tickerPrice_sentimentDummy = df_tickerPrice_sentimentDummy.fillna(0)
print(df_tickerPrice_sentimentDummy)

type(df_final_sample_10q.iloc[0,5])

#for this question we will use df which we built in previous questions, having all of company tickers and their daily return over the period, in addition to rf,rm and rm -rf
print(rm_rf_ticker_return_df)



#to reindex and have date in columns
#we want to join later on date, therefore we need date to be in a same format for both df that we want to join
rm_rf_ticker_return_df=rm_rf_ticker_return_df.reset_index()
rm_rf_ticker_return_df['date of report'] = 0

count =0
#do this to have same type of date to make comparison with other date's column
#just copying date from date col into date of report
for date in rm_rf_ticker_return_df['Date']:
    date = str(date)[:10]
    rm_rf_ticker_return_df.iloc[count,rm_rf_ticker_return_df.shape[1]-1] = date # rm_rf_ticker_return_df.shape[1]  to find last column in which date of report should be inserted
    count +=1

# to buld our regression df, join dummy values to rm_rf_ticker_return_df
regression_rm_rf_ticker_return_df =pd.merge(rm_rf_ticker_return_df, dummy_df, on=['date of report'], how='left')

print(regression_rm_rf_ticker_return_df) # checking our regression df

#feeling zero for dummy that should be zero, thos which are not negative not postive
regression_rm_rf_ticker_return_df =regression_rm_rf_ticker_return_df.fillna(0)
print(regression_rm_rf_ticker_return_df.fillna(0))

#applying regression
#we have 50 stock and around 250 daily return for each of the, an also RM, RF and RM - Rf, to regress we are going to use CAPM equation
#return = alpha = B1*(Market Return - risk free rate) + B2*(Risk free rate) + B3(filling dummy) + e
# to apply this regression for each stock we will  do this regression for all of our sample firms duing the period
#after reporting regression summary we will conduct hypothesis testign which are observable in the output

import statsmodels.formula.api as smf
ticker_sample = df_ticker_price_sample['Ticker'] #assinging sample column to know what firm we had selected randomly
for i in ticker_sample: #iterate over firms
    

    
    try:
        print("########## Regression for : " , i+ " ##############")
        print()
        formula = '{} ~  Rf_Rate + Rm_Rf + sentiment_of_report'.format(i) #using formating to iterate over each firm 
        est_capm = smf.ols(formula, regression_rm_rf_ticker_return_df).fit()
        result = est_capm.summary()
        print(result)
        print()
        print("########## Hypothesis Testing  ##############") #hypothesis testing
        print()



        print("########## Hypothesis Intercept=0  ##############")

        hypothesis01 = 'Intercept=0'
        print(est_capm.t_test(hypothesis01))

        print("########## Hypothesis Rf_Rate=1 ##############")

        hypothesis02 = 'Rf_Rate=1'
        print(est_capm.t_test(hypothesis02))
        print()


        print(est_capm.f_test(hypothesis02))
        print()

        print("########## Hypothesis Rm=1 ##############")


        hypothesis03 = 'Rm=1'
        print(est_capm.t_test(hypothesis03))
        print()


        print(est_capm.f_test(hypothesis03))
        print()


        print("########## Hypothesis Rm_Rf=1 ##############")


        hypothesis04 = 'Rm_Rf=1'
        print(est_capm.t_test(hypothesis04))
        print()


        print(est_capm.f_test(hypothesis04))
        print()


    except:
        print("An exception occurred") #to catch any exceptinn for our record
        continue;










