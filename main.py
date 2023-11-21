import datetime
import math
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning
from sklearn.model_selection import train_test_split


import util

# ========== Settings ==========
needShowSiftedFigures = False
printForecastOut = False

# ================== Подготовка данных ==================
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

train_data = pd.read_excel('Data.xlsx', header=0, index_col='Date', parse_dates=True)
columns = train_data.columns.values

train_data[columns].plot(figsize=(16, 8), subplots=True)
plt.xlabel('Дата')
plt.show()

for column in columns:
    df = train_data[[column]]
    forecast_out = int(math.ceil(0.05 * len(df)))
    if printForecastOut:
        print(forecast_out)
    df = util.extendShiftedData(df, column, forecast_out)
    if needShowSiftedFigures:
        util.showShiftedFigure(df, column)
    scaled_data = util.getScaledData(df)

    #  Выбор данных для прогнозирования
    data_to_be_predicted = scaled_data[-forecast_out:]  # data to be predicted
    data_to_be_trained = scaled_data[:-forecast_out]    # data to be trained

    # Получение целевых значений
    df.dropna(inplace=True)
    shifted = np.array(df['shifted'])

    X_train, X_test, y_train, y_test = train_test_split(data_to_be_trained, shifted, test_size=0.2, random_state=42)
    print('\n=============== Выбор модели для акций: ' + column + ' ===============\n')
    model = util.getModelWithHighAccuracy(X_train, X_test, y_train, y_test)

    last_date = df.index[-1]  # getting the lastdate in the dataset
    last_unix = time.mktime(
        datetime.datetime.strptime(last_date, "%Y.%m.%d").timetuple())  # converting it to time in seconds
    one_day = 86400  # one day equals 86400 seconds
    next_unix = last_unix + one_day  # getting the time in seconds for the next day
    forecast_set = model.predict(data_to_be_predicted)  # predicting forecast data
    df['Forecast'] = np.nan
    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    plt.figure(figsize=(18, 8))
    df[column].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()



