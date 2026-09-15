import numpy as np

import random
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_3x3(self):
        if self.shape != (3,3):
            return
        matrix = Matrix([[0,0,0]
                            ,[0,0,0]
                            ,[0,0,0]])
        for m in range(self.rows):
            for j in range(self.cols):
                minor = Matrix([
                    [self.data[i][k]  for k in range(self.cols) if k != j]
                    for i in range(self.rows) if i != m
                ])
                matrix.data[m][j] = minor.determinant()* (-1)**(m+j)
        adj_matrix = matrix.transpose()
        if self.determinant() == 1:
            print(f"determinant = 1 {adj_matrix}")
            return adj_matrix

        for i in range(self.rows):
            for j in range(self.cols):
                adj_matrix.data[i][j] = adj_matrix.data[i][j] / self.determinant()

        print(f"determinant = {self.determinant()} {adj_matrix}")
        return adj_matrix


    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])



def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])
if __name__ == "__main__":
    """
Task 1:Verify the inverse. Multiply A @ A.inverse_2x2() and confirm you get the identity matrix.
Try it with three different 2x2 matrices. What happens when the determinant is zero?
    """
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    C = Matrix([[3, 4],
                [3, 4]])
    print(C.determinant()) #What happens when the determinant is zero? A: Matrix is singular, no inverse exists!
    print(f"A @ A.inverse_2x2() = {A.matmul(A.inverse_2x2()).data}")
    print(f"B @ B.inverse_2x2() = {B.matmul(B.inverse_2x2()).data}")
    #print(f"C @ C.inverse_2x2() = {C.matmul(C.inverse_2x2()).data}")


    """
Implement 3x3 inverse. Extend the Matrix class to compute inverses for 3x3 matrices using the adjugate method.
Test it against NumPy's np.linalg.inv.
    """
    D = Matrix([[1, 2, 3],
                [0, 1, 4],
                [5, 6, 0]])
    E = Matrix([[1, 2, 3],
                [0, 2, 4],
                [5, 6, 0]])
    D.inverse_3x3()
    E.inverse_3x3()

# Test it against NumPy's np.linalg.inv
    F =[[1, 2, 3],
        [0, 1, 4],
        [5, 6, 0]]
    G =[[1, 2, 3],
        [0, 2, 4],
        [5, 6, 0]]

    print(np.linalg.inv(F))
    print(np.linalg.inv(G))


    """
    Build a two-layer network. Using only your Matrix class (no NumPy),
    create a two-layer neural network: input (3) -> hidden (4) -> output (2)'2 final outputs'.
    Initialize random weights, run a forward pass, and verify all shapes are correct.
    """




    inputs = Matrix([[0.5], [0.8], [0.2]])
    #  a two-layer neural network `hidden (4)`
    w1 = Matrix([
        [random.uniform(-1, 1) for _ in range(3)]
        for _ in range(4)
    ])
    b1 = Matrix([[0.1], [0.1],[0.1], [0.1]])
    hidden_pre = w1.matmul(inputs) + b1
    hidden_output = relu_matrix(hidden_pre) # The 4 activated feature values.

    # Connects 4 hidden features to 2 final answers.
    w2 = Matrix([
        [random.uniform(-1, 1) for _ in range(4)]
        for _ in range(2)
    ])
    b2 =  Matrix([[0.05], [0.05]])





    output_pre = w2.matmul(hidden_output) + b2
    final_output = relu_matrix(output_pre)

    print(f"Input shape:        {inputs.shape}  (Expected: (3, 1))")
    print(f"Weights 1 shape:    {w1.shape}  (Expected: (4, 3))")
    print(f"Bias 1 shape:       {b1.shape}  (Expected: (4, 1))")
    print(f"Hidden Output shape:{hidden_output.shape}  (Expected: (4, 1))")
    print(f"Weights 2 shape:    {w1.shape}  (Expected: (2, 4))")
    print(f"Bias 2 shape:       {b2.shape}  (Expected: (2, 1))")
    print(f"Final Output shape: {final_output.shape}  (Expected: (2, 1))")

    print(f"Final Output Data: {final_output.data}")
