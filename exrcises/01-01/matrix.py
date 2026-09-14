from vector import Vector
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"

    def independent(self):
        ...
    def rank(self):
        ...

    def scaling (self, scaler_x : int ,scaler_y : int ): # TASK : Create a 2D scaling matrix that doubles the x-coordinate and triples the y-coordinate, then apply it to the vector [1, 1]
        list = []
        for i in range(self.shape[0] ):
            count = []
            for j in range(self.shape[1]):
                count.append(self.rows[i][j])
            result_x = count[0] * scaler_x
            result_y = count[1] * scaler_y
            list.append(result_x)
            list.append(result_y)
        return list



if  __name__ == "__main__" :
    rotation_90 = Matrix([[0, -1], [1, 0]])
    point = Vector([3, 1])
    task = Matrix([[1,1]])

    print(task.scaling(3,3))


    rotated = rotation_90 @ point
    print(f"Original: {point}")
    print(f"Rotated 90°: {rotated}")
