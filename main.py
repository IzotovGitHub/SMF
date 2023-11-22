import datetime
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning
from sklearn.model_selection import train_test_split

import Settings
import util
from Settings import directory
from Settings import need_show_sifted_figures
from Settings import print_forecast_out
from Settings import random_state
from Settings import save_to_excel
from Settings import settings as properties
from Settings import shift_percent
from Settings import test_size
from Settings import to_process

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

train_data = pd.read_excel('Data.xlsx', header=0, index_col='Date', parse_dates=True)
columns = train_data.columns.values

if Settings.show_all_history:
    train_data[columns].plot(figsize=(16, 8), subplots=True)
    plt.xlabel('Дата')
    plt.show()

for column in columns:
    settings = properties[column]
    direct = settings[directory]
    if settings[to_process]:
        df = train_data[[column]]
        forecast_out = int(math.ceil(settings[shift_percent] * len(df)))

        if settings[print_forecast_out]:
            print(forecast_out)

        df = util.extend_shifted_data(df, column, forecast_out)

        if settings[save_to_excel]:
            util.save_to_excel(df, direct, 'shifted.xlsx')

        if settings[need_show_sifted_figures]:
            util.show_shifted_figure(df, column)

        scaled_data = util.get_scaled_data(df)

        if settings[save_to_excel]:
            util.save_to_excel(pd.DataFrame(scaled_data), direct, 'scaled_data.xlsx')

        #  Выбор данных для прогнозирования
        '''
            data_to_be_predicted: копия последних данных из scaled_data в количестве равному forecast_out
            data_to_be_trained: копия данных из scaled_data за исключением последних в количестве равному forecast_out
        '''
        data_to_be_predicted = scaled_data[-forecast_out:]
        data_to_be_trained = scaled_data[:-forecast_out]

        if settings[save_to_excel]:
            util.save_to_excel(pd.DataFrame(data_to_be_predicted), direct, 'data_to_be_predicted.xlsx')
            util.save_to_excel(pd.DataFrame(data_to_be_trained), direct, 'data_to_be_trained.xlsx')

        # Получение целевых значений
        df.dropna(inplace=True)  # Remove missing values.
        shifted = np.array(df['shifted'])

        X_train, X_test, y_train, y_test = train_test_split(
            data_to_be_trained,
            shifted,
            test_size=settings[test_size],
            random_state=settings[random_state]
        )

        model = util.get_model_with_high_accuracy(direct, X_train, X_test, y_train, y_test)

        last_date = df.index[-1]
        last_unix = datetime.datetime.strptime(last_date, "%Y.%m.%d").timestamp()
        one_day = 86400
        next_unix = last_unix + one_day
        forecast_set = model.predict(data_to_be_predicted)  # predicting forecast data
        df['Forecast'] = np.nan
        for i in forecast_set:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += one_day
            df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

        df.plot(y=[column, 'Forecast'], figsize=(16, 8))
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()
