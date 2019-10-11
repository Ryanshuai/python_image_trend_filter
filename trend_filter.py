import cvxpy as cp
import numpy as np
import cv2


class Trend_Fiter_2D:
    """
    n: image w or h
    k: derivative order time
    """

    def __init__(self, n, k=0):
        self.n = n
        self.k = k
        self.make_diff_matrix()

    def make_diff_matrix(self):
        up_right_paradiagonal = np.zeros((self.n, self.n))
        for i in range(self.n - 1):
            up_right_paradiagonal[i][i + 1] = 1
        down_left_paradiagonal = up_right_paradiagonal.transpose()

        if self.k == 1:
            up_right_paradiagonal /= 4
            down_left_paradiagonal /= 4

        minus_up_right = np.eye(self.n) - down_left_paradiagonal
        minus_down_left = np.eye(self.n) - up_right_paradiagonal

        self.minus_up = minus_up_right[1:]
        self.minus_down = minus_down_left[:self.n - 1]

        self.minus_left = minus_down_left[:, 1:]
        self.minus_right = minus_up_right[:, :self.n - 1]

        # print(self.minus_up)
        # print(self.minus_down)
        # print(self.minus_left)
        # print(self.minus_right)

    def test_diff(self, mat):
        print('mat_diff_up-----------------------')
        mat_diff_up = np.matmul(self.minus_up, mat)
        print(mat_diff_up)

        print('mat_diff_down-----------------------')
        mat_diff_down = np.matmul(self.minus_down, mat)
        print(mat_diff_down)

        print('mat_diff_left-----------------------')
        mat_diff_left = np.matmul(mat, self.minus_left)
        print(mat_diff_left)

        print('mat_diff_right-----------------------')
        mat_diff_right = np.matmul(mat, self.minus_right)
        print(mat_diff_right)

        return mat_diff_up, mat_diff_down, mat_diff_left, mat_diff_right

    def solve(self, Y, vlambda=50):
        # Define and solve the CVXPY problem.
        X = cp.Variable((self.n, self.n))

        if self.k == 0:
            diff_up = cp.norm(self.minus_up * X, 1)
            diff_down = cp.norm(self.minus_down * X, 1)
            diff_left = cp.norm(X * self.minus_left, 1)
            diff_right = cp.norm(X * self.minus_right, 1)

            obj = 0.5 * cp.sum_squares(Y - X) + vlambda * (diff_up + diff_down + diff_left + diff_right)
        elif self.k == 1:
            kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
            Y_adjacent_mean = cv2.filter2D(Y, -1, kernel / np.sum(kernel))
            obj = 0.5 * cp.sum_squares(Y - X) + vlambda * np.sum(kernel) * cp.norm(X - Y_adjacent_mean, 1)
        else:
            raise ('not imply error')

        prob = cp.Problem(cp.Minimize(obj))
        prob.solve()

        # Print result.
        print("\nThe optimal value is", prob.value)
        return X.value


if __name__ == '__main__':
    # A = np.random.randint(25).reshape(5, 5)
    # A1 = np.diag((1,2,3,4,5))
    # A2 = np.array([[1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1]])
    # A = np.matmul(A1,A2)

    n = 2

    # A = np.random.randint(0, 20, size=n * n).reshape(n, n)
    A = np.arange(0, n*n).reshape(n, n)
    A = np.array([1, 2, 3, 4]).reshape(2, 2)
    print(A)

    tf = Trend_Fiter_2D(n)
    # print(tf.column_shrink)
    mat_diff_up, mat_diff_down, mat_diff_left, mat_diff_right = tf.test_diff(A)

    sum_table = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            up, down, left, right = 0, 0, 0, 0
            if i != 0:
                up = A[i - 1][j] - A[i][j]
            if i != n - 1:
                down = A[i + 1][j] - A[i][j]
            if j != 0:
                left = A[i][j - 1] - A[i][j]
            if j != n - 1:
                right = A[i][j + 1] - A[i][j]
            sum_table[i][j] = abs(up) + abs(down) + abs(left) + abs(right)

    print('sum_table-------------------')
    print(sum_table)
    print(np.sum(sum_table))

    print('mat_diff_sum_table-------------------')
    print(np.sum(np.abs(mat_diff_up))+np.sum(np.abs(mat_diff_down))+np.sum(np.abs(mat_diff_left))+np.sum(np.abs(mat_diff_right)))

