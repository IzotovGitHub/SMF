import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def extend_shifted_data(df: DataFrame, column: str, forecast_out: int):
    df['shifted'] = df[column].shift(-forecast_out)
    return df


def show_shifted_figure(df: DataFrame, column: str):
    df.plot(y=['shifted', column], figsize=(16, 8))
    plt.title('Курс акций: ' + column)
    plt.xlabel('Дата')
    plt.ylabel('Стоимость акции')
    plt.show()


'''
Масштабирование данных
В машинном обучении StandardScaler (стандартный масштабатор) используется для изменения
размера распределения значений так, чтобы среднее значение наблюдаемых значений было равно 0, а стандартное
отклонение – 1
'''


def get_scaled_data(df: DataFrame):
    scaler = StandardScaler()
    X = np.array(df.drop(['shifted'], axis=1))
    return scaler.fit_transform(X)


def get_model_with_high_accuracy(directory: str, X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'Ridge': Ridge(),
        'SVR': SVR()
    }

    score = .0
    method = ''
    idx = 0
    if os.path.exists(directory + 'models_score.xlsx'):
        df = pd.read_excel(directory + 'models_score.xlsx')
        idx = df.index[-1] + 1
    else:
        df = pd.DataFrame(columns=['Linear Regression', 'Random Forest', 'Ridge', 'SVR'])

    for column in df.columns.values:
        sc = get_score(models[column], X_train, X_test, y_train, y_test)
        df.loc[idx, column] = sc
        if sc > score:
            score = sc
            method = column

    save_to_excel(df, directory, 'models_score.xlsx')
    return models[method], method, score


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def save_to_excel(df: DataFrame, directory: str, file_name: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_excel(directory + file_name, index=False)

def save_picture(df: DataFrame, x_lable: str, directory: str, name: str):
    d = directory + 'pictures/'
    if not os.path.exists(d):
        os.makedirs(d)
    df.plot(figsize=(16, 8))
    plt.xlabel(x_lable)
    plt.savefig(d + name)