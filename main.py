import datetime
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ================== Подготовка данных ==================
train_data = pd.read_excel('1.xlsx', header=0, index_col='Date', parse_dates=True)
train_data.info()

# ================== График ==================
train_data.plot(y='Доллар')
plt.title('Dollar History') # adding a title
plt.xlabel('Date') # x label
plt.ylabel('Dollar Price') # y label

#plt.show()

# Копирование столбцов
vtbData = train_data[['VTB']]

forecast_out = int(math.ceil(0.05 * len(vtbData))) # forcasting out 5% of the entire dataset
print(forecast_out)
vtbData['shifted'] = vtbData['VTB'].shift(-forecast_out)

vtbData.plot(y=['shifted', 'VTB'])
plt.title('VTB History') # adding a title
plt.xlabel('Date') # x label
plt.ylabel('VTB Price') # y label
#plt.show()

# Масштабирование данных
# В машинном обучении StandardScaler (стандартный масштабатор) используется для изменения
# размера распределения значений так, чтобы среднее значение наблюдаемых значений было равно 0, а стандартное
# отклонение – 1

scaler = StandardScaler()
X = np.array(vtbData.drop(['shifted'], axis=1))
scaled_data = scaler.fit_transform(X)

#  Выбор данных для прогнозирования
data_to_be_predicted = scaled_data[-forecast_out:] # data to be predicted
data_to_be_trained = scaled_data[:-forecast_out] # data to be trained

# Получение целевых значений
vtbData.dropna(inplace=True)
shifted = np.array(vtbData['shifted'])

X_train, X_test, y_train, y_test = train_test_split(data_to_be_trained, shifted, test_size=0.2, random_state=42)

# Линейная регрессия
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_confidence = lr.score(X_test, y_test)
print('Линейная регрессия')
print(lr_confidence)

# Случайный лес
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_confidence = rf.score(X_test, y_test)
print('Случайный лес')
print(rf_confidence)

# Хребет
rg = Ridge()
rg.fit(X_train, y_train)
rg_confidence = rg.score(X_test, y_test)
print('Хребет')
print(rg_confidence)

# СВР
svr = SVR()
svr.fit(X_train, y_train)
svr_confidence = svr.score(X_test, y_test)
print('СВР')
print(svr_confidence)

names = ['Linear Regression', 'Random Forest', 'Ridge', 'SVR']
columns = ['model', 'accuracy']
scores = [lr_confidence, rf_confidence, rg_confidence, svr_confidence]
alg_vs_score = pd.DataFrame([[x, y] for x, y in zip(names, scores)], columns = columns)

last_date = vtbData.index[-1] #getting the lastdate in the dataset
last_unix = time.mktime(datetime.datetime.strptime(last_date, "%Y.%m.%d").timetuple()) #converting it to time in seconds
one_day = 86400 #one day equals 86400 seconds
next_unix = last_unix + one_day # getting the time in seconds for the next day
forecast_set = lr.predict(data_to_be_predicted) # predicting forecast data
vtbData['Forecast'] = np.nan
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    vtbData.loc[next_date] = [np.nan for _ in range(len(vtbData.columns)-1)]+[i]


plt.figure(figsize=(18, 8))
vtbData['VTB'].plot()
vtbData['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

