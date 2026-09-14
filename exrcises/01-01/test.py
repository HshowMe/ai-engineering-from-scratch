def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    for i in range(len(matrix)):
        if len(matrix[i]) == 0:
            matrix[i] *= scalar
            return matrix
        else:
            for j in range(len(matrix[i])):
                matrix[i][j] *= scalar
    return matrix




scalar_multiply([[1]], 0)
