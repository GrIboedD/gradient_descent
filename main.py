import numpy as np


def simple_gradient(function, parameters, argument_increment = 1e-06):
    """
    Алгоритм для вычисления градиента с помощью определения производной
    :param function: исходная функция
    :param parameters: параметры функции
    :param argument_increment: значение приращения аргумента (не обязательно)
    :return:
    """
    gradient = np.array([])
    for i in range(len(parameters)):
        incremental_parameters = parameters.copy()
        incremental_parameters[i] += argument_increment
        gradient = np.append(gradient, (function(*incremental_parameters) - function(*parameters)) / argument_increment)
    return gradient



def gradient_descent(function, gradient, initial_parameters, learn_rate, n_iter, tolerance = 1e-06):
    """
    Алгоритм градиентного спуска для поиска локального минимума функции
    :param function: функция, минимум которой необходимо найти
    :param gradient: функция расчета градиента
    :param initial_parameters: начальные значения параметров
    :param learn_rate: скорость спуска
    :param n_iter: количество итераций
    :param tolerance: значение для остановки алгоритма, когда изменение по каждому параметру <= значению (не обязательно)
    :return: значение параметров в предполагаемом минимуме функции
    """
    parameters = initial_parameters.copy()
    for _ in range(n_iter):
        difference = learn_rate * gradient(function, parameters)
        if np.all(np.abs(difference) <= tolerance):
            break

        parameters -= difference

    return parameters


if __name__ == '__main__':
    print(gradient_descent(lambda x: x ** 2, simple_gradient, [5], 0.5, 100))
