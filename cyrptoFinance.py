import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
#use svm to calculate regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

quandl.ApiConfig.api_key = 'SJ9U6zFRerxKsjgZYEnx'

style.use('ggplot')

df = quandl.get("BCHARTS/KRAKENUSD", start_date="2017-01-01")
#start = dt.datetime(2000,1,1)
#end = dt.datetime(2016,12,31)
df.to_csv('BITCOIN.csv')
df = pd.read_csv('BITCOIN.csv', parse_dates = True, index_col = 0)

df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close', 'HL_PCT', 'PCT_change','Volume (BTC)']]

forecast = 'Close'
df.fillna(-99999, inplace = True)
print(df.tail())

outForecast = int(math.ceil(0.01*len(df)))


df['label'] = df[forecast].shift(-outForecast)

x = np.array(df.drop(['label'],1))
x = preprocessing.scale(x)
x = x[:-outForecast]
x_lately = x[-outForecast:]

df.dropna(inplace=True)
y = np.array(df['label'])
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.9)
clf = LinearRegression(n_jobs = -1)
clf.fit(x_train, y_train)

with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
    
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)


accuracy = clf.score(x_test, y_test)

forecast_set = clf.predict(x_lately)
print(forecast_set, accuracy, outForecast)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
    
print(df.tail())

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


#print(accuracy)
             
##def process_data_for_labels(ticker):
##    hm_days = 7
##    df = pd.read_csv('BITCOIN.csv', parse_dates = True, index_col = 0)
    
