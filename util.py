import numpy as np
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
    plt.title('Курс акций: ' + column)  # adding a title
    plt.xlabel('Дата')  # x label
    plt.ylabel('Стоимость акции')  # y label
    plt.show()


# Масштабирование данных
# В машинном обучении StandardScaler (стандартный масштабатор) используется для изменения
# размера распределения значений так, чтобы среднее значение наблюдаемых значений было равно 0, а стандартное
# отклонение – 1
def get_scaled_data(df: DataFrame):
    scaler = StandardScaler()
    X = np.array(df.drop(['shifted'], axis=1))
    return scaler.fit_transform(X)


def get_model_with_high_accuracy(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'Ridge': Ridge(),
        'SVR': SVR()
    }

    score = .0
    method = ''
    for mtd in models:
        sc = get_score(models[mtd], X_train, X_test, y_train, y_test)
        print(mtd + ' score: ' + str(sc))
        if sc > score:
            score = sc
            method = mtd
    print('\nВыбрана наилучшая модель: ' + method + '\nС точностью: ' + str(score))
    return models[method]


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)
