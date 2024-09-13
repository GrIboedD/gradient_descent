def parabola(x):
    return (x + 10) ** 2


def my_function(x, y):
    return x ** 2 + y ** 2


def simple_gradient_descent(function, parameters, step, eps1, eps2):
    """
    Градиентный спуск для поиска локального минимума функции
    :param function: исходная функция
    :param parameters: начальные значения параметров
    :param step: скорость спуска
    :param eps1: максимальное значение частной производной для остановки алгоритма
    :param eps2: значения приращения аргумента для вычисления частных производных
    :return: значение параметров при локальном минимуме
    """
    while True:
        gradient = []
        for i in range(len(parameters)):
            tmp_parameters = parameters.copy()
            tmp_parameters[i] += eps2
            gradient.append((function(*tmp_parameters) - function(*parameters)) / eps2)

        is_end = True
        for i in range(len(gradient)):
            if abs(gradient[i]) > eps1:
                is_end = False

        if is_end:
            break

        for i in range(len(parameters)):
            parameters[i] -= gradient[i] * step

    return parameters


if __name__ == '__main__':
    print(simple_gradient_descent(parabola, [-5], 0.1, 0.001, 0.0001))
