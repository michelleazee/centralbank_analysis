# Fed Funds Rate Prediction

## Introduction

This model utilizes economic indicators and text data from the FOMC to predict Fed Funds Rate decisions from 1 - 18 months into the future. Data is gathered and cleaned using code from Yuki Takahashi's FedSpeak project (description below). In Notebooks 5, I add LDA topic predictions as text data. Together with the LDA features, a RandomForest Classifier is used to make predictions on whether the Fed would increase, decrease, or hold their rates in the next 12 scheduled meetings (approximiately 18 months as meetings are every six weeks).


## FedSpeak — How to build a NLP pipeline to predict central bank policy changes

## Table of Contents
1. Project Description
2. Installation
3. Data Understanding
4. Code Description
5. Licensing, Authors, Acknowledgements

## 1. Project Description
Posted a Medium Blog here:
https://yuki678.medium.com/fedspeak-how-to-build-a-nlp-pipeline-to-predict-central-bank-policy-changes-a2f157ca0434?sk=989433349aed4e6dd1faf5a72e848e35

Please refer to the post for business understanding, project overview and analysis result.

## 2. Installation
#### Libraries
Required libraries are described in requirements.txt. The code should run with no issues using Python versions 3.6+.
Create a virtual environment of your choice. Here uses Anaconda:
```
conda create -n fomc python=3.6 jupyter
conda activate fomc
pip install -r requirements.tx
```
#### Download input data
1. Create data directory
   ```
   cd data
   mkdir FOMC MarketData LoughranMcDonald GloVe preprocessed train_data result
   cd FOMC
   mkdir statement minutes presconf_script meeting_script script_pdf speech testimony chair
   cd ../MarketData
   mkdir Quandl
   ```
2. Move to src directory
   `cd ../../src`
3. Get data from FOMC Website. Specify document type. You can also specify from year.
   `python FomcGetData.py all 1980`
4. Get calendar from FOMC Website. Specify from year.
   `python FomcGetCalendar.py 1980`
5. Get data from Quandl. Specify your API Key and From Date (yyyy-mm-dd). You can specify Quandl Code, otherwise all required data are downloaded.
   `python QuandlGetData.py [your API Key] 1980-01-01`
6. Download Sentiment Dictionary in data/LoughranMcDonald directory in csv
   * Loughran and McDonald Sentiment Word Lists (https://sraf.nd.edu/textual-analysis/resources/) 

#### To Run Notebook (Local)

1. Go to top directory
   `cd ../`
2. Run the jupyter notebooks 
   `jupyter notebook`
3. Open and run notebooks No.1 to No.8 for analysis

#### To Run Notebook (Google Colab)
All notebooks can be executed on Google Colab. 
1. Upload notebooks to your Google Drive
2. Upload downloaded data to your Google Drive (Colab Data dir)
3. Execute each notebook (Note: You need to authorize the access to your Google Drive when asked to input the code)


## 3. Data Understanding
Text data is scraped from FOMC Website. Other economic and market data are downloaded from FRB of St. Louis website (FRED)
Data used for each prediction are only those available before the meeting.

#### Text Data
* FOMC/fomc_calendar.pickle - all FOMC calendar dates
* FOMC/statement.pickle - Statement text along with basic attributes such as dates, speaker, title. Each text is also available in the directory with the same name. Statements are available post press conference for almost all meetings, which include rate decision and target rate. From 2008, target rate became a range instead of a single value.
* FOMC/minutes.pickle - Minutes text along with basic attributes such as dates, speaker, title. Each text is also available
 in the directory with the same name. Minutes are summary of FOMC Meeting and contents are structured in sections and paragraphs, most of which were updated in 2011 and 2012. The minutes of regularly scheduled meetings are released three weeks after the date of the policy decision.
* FOMC/presconf_script.pickle - Press conference scripts text along with basic attributes such as dates, speaker, title. Each text is also available in the directory with the same name. This is available from 2011. Starting with the speaker name, so extract those spoken by the chairperson because the other person's words are more likely to be questions and not FOMC's view. It is in pdf form, so download pdf and then process the text.
* FOMC/meeting_script.pickle - Meeting scripts text along with basic attributes such as dates, speaker, title. Each text is also available in the directory with the same name. FOMC decided to publish this five years after each meeting. It contains all the words spoken during the meeting. It will contain some insight about FOMC discussions and how the consensus about monetary policy is built, but cannot be used in prediction as this is not published for five years.  It is in pdf form, so download pdf and then process the text.
* FOMC/speech.pickle -  Speech text along with basic attributes such as dates, speaker, title. Each text is also available in the directory with the same name. There are many speeches published but some of them are not related to monetary policies but various topics such as regulations and governance. Some speeches may contain indication of FOMC policy, so use only those by the chairperson.
* FOMC/testimony.pickle -  Testimony text along with basic attributes such as dates, speaker, title. Each text is also available in the directory with the same name. Like speeches, testimony is not necessarily related to monetary policy. There are semi-annual testimony in the congress, which can be a good inputs of FOMC's view by chairperson, so use only those by the chairperson.

#### Market Data
In MarketData/Quandl, csv is saved with Quandl Code as the file name.
* FED Rate
  * FRED_DFEDTAR.csv - Target FED Rate till 2008, Daily
  * FRED_DFEDTARU.csv - Target Upper FED Rate from 2008, Daily
  * FRED_DFEDTARL.csv - Target Lower FED Rate from 2008, Daily
  * FRED_DFF.csv - Effective FED Rate, Daily
* GDP
  * FRED_GDPC1.csv - Real GDP, Quarterly
  * FRED_GDPPOT.csv - Real potential GDP, Quarterly
* CPI
  * FRED_PCEPILFE.csv - Core PCE excluding Food and Energy, Monthly
  * FRED_CPIAUCSL.csv - Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
* Employment
  * FRED_UNRATE.csv - Unemployment Rate, Monthly
  * FRED_PAYEMS.csv - Employment, Monthly
* Sales
  * FRED_RRSFS.csv - Advance Real Retail and Food Services Sales, monthly
  * FRED_HSN1F.csv - New Home Sales, monthly
* ISM
  * ISM_MAN_PMI.csv - ISM Purchasing Managers Index
  * ISM_NONMAN_NMI.csv - ISM Non-manufacturing Index
* Treasury
  * USTREASURY_YIELD.csv - This is optional as not used in the final analysis.

#### Loughran-McDonald Dictionary
* LoughranMcDonald/LoughranMcDonald_SentimentWordLists_2018.csv - This is used in preliminary analysis and creating Tfidf vectors.

## 4. Code Description
#### 1_FOMC_Analysis_Preliminary.ipynb
First, take a glance at the FOMC statement to see if it contains any meaningful information.
##### Input: 
* ../data/FOMC/statement.pickle
* ../data/MarketData/Quandl/FRED_DFEDTAR.csv
* ../data/MarketData/Quandl/FRED_DFEDTARU.csv
* ../data/MarketData/Quandl/FRED_DFEDTARL.csv
##### Output:
* None
##### Process:
1. Analyze sentiment of the statement text using Loughran and McDonald Sentiment Word Lists
2. Plot sentiment (count of positive words with negation, negative words and net over time series, normalized by the number of words
3. Load FED Rate, map the rate and decision to statement
4. Plot the moving average of the sentiment along with FED rate and recession period
5. Plot the same with Quantitative Easing and Chairpersons

#### 2_FOMC_Analysis_Preprocess_NonText.ipynb
Next, preprocess nontext meta data. Do necessary calculations and add to the calendar dataframe to map those latest available indices as input to the FOMC Fed rate decision.

##### Input: 
* ../data/FOMC/fomc_calendar.pickle
* All Market Data and Economic Indices
##### Output:
* ../data/preprocessed/nontext_data
* ../data/preprocessed/nontext_ma2
* ../data/preprocessed/nontext_ma3
* ../data/preprocessed/nontext_ma6
* ../data/preprocessed/nontext_ma12
* ../data/preprocessed/treasury
* ../data/preprocessed/fomc_calendar
##### Process:
1. Load and plot all numerical data
2. Add FED Rate and rate decisions to FOMC Meeting Calendar
3. Add QE as Lowering event and Tapering as Raising event
4. Add the economic indices to the FOMC Meeting Calendar
5. Calculate Taylor rule
6. Calculate moving average
7. Save data

#### 3_FOMC_Analysis_Preprocess_Text.ipynb
##### Input: 
* ../data/preprocessed/fomc_calendar.pickle
* ../data/FOMC/statement.pickle
* ../data/FOMC/minutes.pickle
* ../data/FOMC/meeting_script.pickle
* ../data/FOMC/presconf_script.pickle
* ../data/FOMC/speech.pickle
* ../data/FOMC/testimony.pickle
##### Output: 
* ../data/preprocessed/text_no_split
* ../data/preprocessed/text_split_200
* ../data/preprocessed/text_keyword

##### Process: 
1. Add QE announcement to statement
2. Add Rate and Decision to Statement, Minutes, Meeting Script and Presconf Script
3. Add Word Count, Next Meeting Date, Next Meeting Rate and Next Meeting Decision to all inputs
4. Remove return code and separate text by sections
5. Remove short sections - having less number of words that threshold as it is unlikely to hold good information
6. Split text of Step 5 to maximum of 200 words with 50 words overlap
7. Filter text of Step 5 for those having keyword at least 2 times only

#### 4_FOMC_Analysis_EDA_FE_NonText.ipynb
##### Input: 
* ../data/preprocessed/nontext_data.pickle
* ../data/preprocessed/nontext_ma2.pickle
* ../data/preprocessed/nontext_ma3.pickle
* ../data/preprocessed/nontext_ma6.pickle
* ../data/preprocessed/nontext_ma12.pickle
##### Output: 
* ../data/train_data/nontext_train_small
* ../data/train_data/nontext_train_large

##### Process: 
1. Check correlation to find good feature to predict Rate Decision
2. Check correlation of moving average to Rate Decision
3. Check correlation of calculated rates and changes by taylor rules
4. Compare distribution of each feature between Rate Decision
5. Fill missing values
6. Create small dataset with selected 9 features and large dataset, which contains all


### Other Files
* FomcGetCalendar.py - From FOMC Website, create fomc_calendar to save in pickle and csv
* FomcGetData.py - Calls relevant classes to get data from FOMC Website
* QuandlGetData.py - Get market data from Quandl.
* fomc_get_data/FomcBase.py - Base abstract class to scrape FOMC Website to download text data
* fomc_get_data/FomcStatement.py - Child class of FomcBase to retrieve statement texts
* fomc_get_data/FomcMinutes.py - Child class of FomcBase to retrieve minutes texts
* fomc_get_data/FomcPresConfScript.py - Child class of FomcBase to retrieve press conference script texts
* fomc_get_data/FomcMeetingScript.py - Child class of FomcBase to retrieve meeting script texts
* fomc_get_data/FomcSpeech.py - Child class of FomcBase to retrieve speech texts
* fomc_get_data/FomcTestimony.py - Child class of FomcBase to retrieve testimonny texts

The followings are used only for initial check and not required to run:
* FOMC_analyse_website.ipynb
* FOMC_analyse_website_2.ipynb
* FOMC_check_FEDRate.ipynb
* FOMC_Analysis_BERT_MultiSampleDropoutModel.ipynb
* FOMC_Analysis_BERT_Tensorflow.ipynb
* FOMC_Post_Training_BERT.ipynb
* FOMC_Text_Summarization.ipynb

## 5. Licensing, Authors, Acknowledgements
Data attributes to the source (FRED, ISM, US Treasury and Quandl). Loughran McDonald dictionary attributes to https://sraf.nd.edu/textual-analysis/resources/ in University of Notre Dame.
Feel free to use the source code as you would like!
![image](https://user-images.githubusercontent.com/71299641/130502190-7dee448b-2900-4064-b647-aeb5c814db1d.png)
