import numpy as np


def solve_linear_system(A, b):

    # Проверка на совместность системы
    if np.linalg.matrix_rank(A) != np.linalg.matrix_rank(np.hstack((A, b.reshape(-1, 1)))):
        raise ValueError("Система уравнений несовместна или имеет бесконечное количество решений.")

    # Решение системы уравнений
    x = np.linalg.solve(A, b)
    return x


# Пример использования
A = np.array([[2, 1, -1],
              [3, 4, 2],
              [1, -5, -2]])

b = np.array([3, 7, -5])

solution = solve_linear_system(A, b)
print("Решение СЛАУ:", solution)

# Примеры СЛАУ
example_1_A = np.array([[7, 13, -23],
                        [17, 4, 2],
                        [1, -25, -2]])

example_1_b = np.array([3, 8, -12])

example_2_A = np.array([[7, 4, 3],
                        [15, 11, -2],
                        [22, 3, -3]])

example_2_b = np.array([6, 5, 9])

# Тестирование
try:
    print("Решение 1:")
    print("Матрица коэффициентов:")
    print(example_1_A)
    print("Вектор правой части:")
    print(example_1_b)
    solution_1 = solve_linear_system(example_1_A, example_1_b)
    print("Решение СЛАУ:", solution_1)
except ValueError as e:
    print("Произошла ошибка:", e)

print("\n")

try:
    print("Решение 2:")
    print("Матрица коэффициентов:")
    print(example_2_A)
    print("Вектор правой части:")
    print(example_2_b)
    solution_2 = solve_linear_system(example_2_A, example_2_b)
    print("Решение СЛАУ:", solution_2)
except ValueError as e:
    print("Произошла ошибка:", e)

