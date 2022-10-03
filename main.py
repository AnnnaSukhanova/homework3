import numpy as np
import random
from more_itertools import powerset
import plotly.express as px
import plotly
from plotly.graph_objs import Scatter
from visualisation import *


class Functor:
    def __init__(self, function, function_name):
        self.__function = function
        self.__function_name = function_name

    def __call__(self, x):
        return self.__function(x)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.__function_name

    @property
    def name(self):
        return self.__function_name


class RegularizationRegression:
    a_array = [0., 0.01, 0.001, 0.1, 0.5, 1., 5, 10, 50, 100]
    base_functions = [
        Functor(lambda x: np.sin(6 * np.pi * x), "sin(6*pi*x)"),
        Functor(lambda x: np.cos(x), "cos(x)"),
        Functor(lambda x: np.exp(x), "exp(x)"),
        Functor(lambda x: x, "x"),
    ]
    amount_of_base = 10

    @staticmethod
    def build_design_matrix(functions, X):
        result = np.ones((X.size, len(functions) + 1))
        for j in range(1, len(functions) + 1):
            result[:, j] = functions[j - 1](X)
        return result

    @staticmethod
    def make_power(functor, power):  # добавляем x^power в список базисных функций
        return Functor(lambda x: functor(x) ** power, f"({functor.name})^{power}")

    @staticmethod
    def make_polynoms(f):  # получаем  x^(от 2 до 9), x^(от 10 до 100 c шагом 10)
        result = []
        for i in range(2, 10):
            result.append(RegularizationRegression.make_power(f, i))
        for i in range(10, 101, 10):
            result.append(RegularizationRegression.make_power(f, i))
        return result

    @staticmethod
    def expand_functions(base_functions):  # обновляем список базисных функций
        result = []
        for i in range(0, 4):
            result.append(base_functions[i])
        result.extend(
            RegularizationRegression.make_polynoms(base_functions[3])
        )
        return result

    @staticmethod
    def pick_functions(base_functions):  # выбираем функции
        result = random.sample(list(powerset(base_functions))[23:35443], 30, counts=None)
        return result

    @staticmethod
    def learning(design_matrix, T, a):
        return np.linalg.pinv(
            design_matrix.T @ design_matrix
            + a * np.eye(design_matrix.T.shape[0])
        ) @ design_matrix.T @ T

    @staticmethod
    def calculate_error(t, W, design_matrix):
        return (1 / 2) * sum((t - (W @ design_matrix.T)) ** 2)


def train_test_validation_split(X, t):
    ind_prm = np.random.permutation(np.arange(N))  # перемешивает данные, это индексы наших массивов
    tr = 0.8
    val = 0.1
    train_ind = ind_prm[:int(tr * N)]  # до 0,8
    valid_ind = ind_prm[int(tr * N):int((val + tr) * N)]  # от 0,8 до 0,9
    test_ind = ind_prm[int((val + tr) * N):]  # от 0,9 до 1
    x_train, x_valid, x_test = X[train_ind], X[valid_ind], X[test_ind]
    t_train, t_valid, t_test = t[train_ind], t[valid_ind], t[test_ind]
    return x_train, t_train, x_valid, t_valid, x_test, t_test


def united(weight, func):
    functions = []
    for i in range(0, len(func)):
        str = f"y = {round(weight[i][0], 2)}"
        for j in range(1, len(func[i]) + 1):
            str += f" + ({round(weight[i][j], 2)}){func[i][j - 1]}"
        functions.append(str)
    return np.array(functions)


def test_regularization(X, t, Z):
    x_train, t_train, x_valid, t_valid, x_test, t_test = train_test_validation_split(X, t)

    all_functions = RegularizationRegression.expand_functions(
        RegularizationRegression.base_functions)
    functions = RegularizationRegression.pick_functions(all_functions)
    new_a = random.sample(RegularizationRegression.a_array, 5, counts=None)
    data = np.empty((150, 5), dtype="object")
    counter = -1
    for a in new_a:
        counter += 1
        for i in range(0, 30):
            function = functions[i]
            F = RegularizationRegression.build_design_matrix(function, x_train)
            W = RegularizationRegression.learning(F, t_train, a)
            validate_F = RegularizationRegression.build_design_matrix(function, x_valid)
            E = RegularizationRegression.calculate_error(t_valid, W, validate_F)
            test_design = RegularizationRegression.build_design_matrix(function, x_test)
            test_error = RegularizationRegression.calculate_error(t_test, W, test_design)
            data[counter * 30 + i][0] = function
            data[counter * 30 + i][1] = E
            data[counter * 30 + i][2] = W
            data[counter * 30 + i][3] = test_error
            data[counter * 30 + i][4] = a
    counter = 1
    while counter < 150:  # сортировка получившегося массива с данными по валидационной ошибке
        for i in range(150 - counter):
            if data[i][1] > data[i + 1][1]:
                data[[i, i + 1]] = data[[i + 1, i]]
        counter += 1
    full_design = RegularizationRegression.build_design_matrix(data[0][0], X)

    validation = data[0:10, 1]
    test = data[0:10, 3]
    func = data[0:10, 0]
    a = data[0:10, 4]
    weight = data[0:10, 2]
    func2 = united(weight, func)
    best_test = round(data[0][3], 2)
    best_y = func2[0]

    visualisation = Visualization()
    visualisation.models_error_scatter_plot(validation, test, func2, a,
                                            title='10 лучших моделей',
                                            show=True,
                                            save=True,
                                            name="10_best_func",
                                            path2save="C:/plotly")

    df = px.data.tips()
    fig = px.scatter(df, x=X, y=t, opacity=0.65, title=f"test error = {best_test},  лямбда = {data[0][4]}, {best_y}")
    fig.add_traces(Scatter(x=X, y=Z, name='z(x)'))
    fig.add_traces(Scatter(x=X, y=data[0][2] @ full_design.T, name='y(x,w)'))
    plotly.offline.plot(fig, filename=f'C:/plotly/best_model.html')


N = 1000
X = np.linspace(0, 1, N)
Z = 20 * np.sin(2 * np.pi * 3 * X) + 100 * np.exp(X)
error = 10 * np.random.randn(N)
t = Z + error

test_regularization(X, t, Z)
