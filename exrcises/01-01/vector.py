#Task 1
import random
import math
import numpy as np

class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def project_onto(self, other):
        scalar = self.dot(other) / other.dot(other)
        return Vector([scalar * x for x in other.components])

    def __repr__(self):
        return f"Vector({self.components})"

    def angle_between(self, other):
        a = Vector(self.components)
        b = Vector(other.components)
        product = a.dot(b)
        magni = a.magnitude() * b.magnitude()
        if magni == 0:
            return 0
        result = product / magni
        #result = max(-1.0, min(1.0, result))
        angle_radians = math.degrees(math.acos(result))

        return angle_radians




def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = w.project_onto(u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal

def check_orthonormal(vectors):
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if vectors[i].dot(vectors[j]) != 0:
                return False
        if vectors[i].magnitude() != 1:
            return False
    return True

if __name__ == "__main__":
    a = Vector([1, 2, 3])
    b = Vector([4, 5, 6])

    print(f"angle is {a.angle_between(b)}")
    print(f"a + b = {a + b}")
    print(f"a · b = {a.dot(b)}")
    print(f"|a| = {a.magnitude():.4f}")
    print(f"cosine similarity = {a.cosine_similarity(b):.4f}")



    # Task 3
    n = 5
    dim = 50
    random.seed(42)
    rand_vectors= []


    for _ in range(n):
        v = [random.uniform(-1.0, 1.0) for _ in range(dim)]
        rand_vectors.append(Vector(v))
    compare = []
    for i in range(n):
        for j in range(i + 1, n):
            res = rand_vectors[i].cosine_similarity(rand_vectors[j])
            print(f"vector {i+1} has been compared with vector {j+1}, result is : {res}")
            compare.append((res,i+1,j+1))


    max_res, best_i, best_j = max(compare, key=lambda x: x[0])


    print(f"most similar two vectors are v{best_i} and v{best_j} with value of : {max_res}")


    # Task 4
    u1 = Vector([1, 1, 0])
    u2 = Vector([1, 0, 1])
    u3 = Vector([0, 1, 1])
    orthonormal = gram_schmidt([u1, u2, u3])
    check = check_orthonormal(orthonormal)
    print(f"orthonormal = {orthonormal}")
    print(f"check = {check}")

    #Task 5
    a = [[1, 1, 0],
         [1, 0, 1],
         [0, 1, -1]]

    print (np.linalg.matrix_rank(a))

    c = np.array([1, 2 ,3])
    d = np.array([1, 1 ,1])
    proj = (np.dot(c, d) / np.dot(d, d)) * d
    print(f"Projection of {c} onto {d}: {proj}")
