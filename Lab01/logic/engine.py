from math import sqrt
from copy import deepcopy


def compute_determinant(matrix: list[list[float]]) -> float:
    return (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
            matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
            matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))


def compute_trace(matrix: list[list[float]]) -> float:
    return sum(matrix[x][x] for x in range(0, 3))


def compute_vector_norm(vector: list[float]) -> float:
    return sqrt(sum([x ** 2 for x in vector]))


def compute_transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def compute_matrix_vector_multiplication(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(map(lambda x: x[0] * x[1], zip(line, vector))) for line in matrix]


def replace_matrix_column(matrix: list[list[float]], vector: list[float], column_index: int) -> list[list[float]]:
    new_matrix = deepcopy(matrix)
    for i in range(0, 3):
        new_matrix[i][column_index] = vector[i]
    return new_matrix


def solve_via_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    determinant = compute_determinant(matrix)

    x_matrix = replace_matrix_column(matrix, vector, 0)
    x_determinant = compute_determinant(x_matrix)

    y_matrix = replace_matrix_column(matrix, vector, 1)
    y_determinant = compute_determinant(y_matrix)

    z_matrix = replace_matrix_column(matrix, vector, 2)
    z_determinant = compute_determinant(z_matrix)

    return [result / determinant for result in [x_determinant, y_determinant, z_determinant]]


def compute_cofactor(line: int, column: int, matrix: list[list[float]]) -> float:
    sign = (-1) ** (line + column)

    minor = []
    for i in range(0, 3):
        if i == line:
            continue
        minor.append([])
        for j in range(0, 3):
            if j == column:
                continue
            minor[-1].append(matrix[i][j])

    minor_determinant = minor[0][0] * minor[1][1] - minor[0][1] * minor[1][0]

    return minor_determinant * sign


def compute_cofactor_matrix(matrix: list[list[float]]) -> list[list[float]]:
    new_matrix = []

    for i in range(0, 3):
        new_matrix.append([])
        for j in range(0, 3):
            new_matrix[-1].append(compute_cofactor(i, j, matrix))

    return new_matrix


def compute_adjugate(matrix: list[list[float]]) -> list[list[float]]:
    cofactor_matrix = compute_cofactor_matrix(matrix)
    return compute_transpose(cofactor_matrix)


def compute_inverse(matrix: list[list[float]]) -> list[list[float]]:
    determinant = compute_determinant(matrix)
    assert determinant != 0

    adjugate = compute_adjugate(matrix)
    return list(map(lambda x: list(map(lambda y: y / determinant, x)), adjugate))


def solve_via_inversion(matrix: list[list[float]], vector: list[float]) -> list[float]:
    inverse = compute_inverse(matrix)
    return compute_matrix_vector_multiplication(inverse, vector)
