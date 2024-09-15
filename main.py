import numpy as np


def loss_function_simple_linear_regression(a, b, x_arr, y_arr):
    """
    Функция ошибки для простой линейной регрессии
    :param a: параметр a
    :param b: параметр b
    :param x_arr: массив входных данных
    :param y_arr: массив выходных данных
    :return: сумму квадратов разности расчетных и наблюдаемых значений, деленную на 2N
    """
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    ssr = (y_arr - a - b * x_arr) ** 2
    return np.sum(ssr) / (2 * len(y_arr))


def simple_gradient(function, parameters, input_data=None, output_data=None, argument_increment=1e-06):
    """
    Алгоритм для вычисления градиента с помощью определения производной
    :param function: исходная функция
    :param parameters: параметры функции
    :param input_data: матрица входных данных
    :param output_data: матрица выходных данных
    :param argument_increment: значение приращения аргумента (не обязательно)
    :return:
    """
    gradient = []
    for i in range(len(parameters)):
        incremental_parameters = parameters.copy()
        incremental_parameters[i] += argument_increment

        if input_data is None and output_data is None:
            gradient.append((function(*incremental_parameters) - function(*parameters)) / argument_increment)
        else:
            gradient.append((function(*incremental_parameters, input_data, output_data) - function(
                *parameters, input_data, output_data)) / argument_increment)
    return gradient


def gradient_descent(loss_function, gradient, initial_parameters, learn_rate, n_iter, input_data=None, output_data=None,
                     tolerance=1e-06):
    """
    Алгоритм градиентного спуска для поиска локального минимума функции
    :param loss_function: функция ошибки, минимум которой необходимо найти
    :param gradient: функция расчета градиента
    :param initial_parameters: начальные значения параметров
    :param learn_rate: скорость спуска
    :param n_iter: количество итераций
    :param input_data: матрица входных данных (не обязательно)
    :param output_data: матрица выходных данных (не обязательно)
    :param tolerance: значение для остановки алгоритма, когда изменение по каждому параметру <= значению (не обязательно)
    :return: значение параметров в предполагаемом минимуме функции
    """
    parameters = initial_parameters.copy()
    for _ in range(n_iter):

        if input_data is None and output_data is None:
            vector = np.array(gradient(loss_function, parameters))
        else:
            vector = np.array(gradient(loss_function, parameters, input_data, output_data))

        difference = learn_rate * vector

        if np.all(np.abs(difference) <= tolerance):
            break

        parameters -= difference

    return parameters


def stochastic_gradient_descent(loss_function, gradient, initial_parameters, learn_rate, decay_rate, n_iter, input_data,
                                output_data, batch_size=1,
                                tolerance=1e-06):
    """
    Алгоритм стохастического градиентного спуска для поиска локального минимума функции
    :param loss_function: функция ошибки, минимум которой необходимо найти
    :param gradient: функция расчета градиента
    :param initial_parameters: начальные значения параметров
    :param learn_rate: скорость спуска
    :param decay_rate: инерция спуска
    :param n_iter: количество итераций
    :param input_data: матрица входных данных
    :param output_data: матрица выходных данных
    :param batch_size: размер массива случайно выбранных наблюдаемых данных для расчета градиента (не обязательно)
    :param tolerance: значение для остановки алгоритма, когда изменение по каждому параметру <= значению (не обязательно)
    :return: значение параметров в предполагаемом минимуме функции
    """
    parameters = initial_parameters.copy()
    batch_number = len(input_data) // batch_size
    difference = 0
    for _ in range(n_iter):
        indexes = np.random.permutation(len(input_data))
        shuffled_input = np.array(input_data)[indexes]
        shuffled_output = np.array(output_data)[indexes]
        start = 0
        for _ in range(batch_number):
            batch_input = shuffled_input[start: start + batch_size]
            batch_output = shuffled_output[start: start + batch_size]
            start = start + batch_size
            vector = np.array(gradient(loss_function, parameters, batch_input, batch_output))
            difference = learn_rate * vector - difference * decay_rate
            if np.all(np.abs(difference) <= tolerance):
                break
            parameters -= difference

    return parameters


if __name__ == '__main__':
    print(gradient_descent(lambda u, v: u ** 2 + v ** 2, simple_gradient, [5, 5], 0.5, 100))

    x = [5, 15, 25, 35, 45, 55]
    y = [5, 20, 14, 32, 22, 38]

    print(gradient_descent(loss_function_simple_linear_regression, simple_gradient, [0.5, 0.5], 0.0008, 100_000, x, y))

    print(stochastic_gradient_descent(loss_function_simple_linear_regression, simple_gradient, [0.5, 0.5], 0.0001, 0.3,
                                      100_000, x, y, 3))
