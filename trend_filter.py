
import cvxpy as cp
import numpy as np


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
        for i in range(self.n-1):
            up_right_paradiagonal[i][i+1] = 1
        down_left_paradiagonal = up_right_paradiagonal.transpose()

        minus_up_right = np.eye(self.n)-down_left_paradiagonal
        minus_down_left = np.eye(self.n)-up_right_paradiagonal

        self.minus_up = minus_up_right[1:]
        self.minus_down = minus_down_left[:self.n-1]

        self.minus_left = minus_down_left[:, 1:]
        self.minus_right = minus_up_right[:, :self.n-1]

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

    def solve(self, Y, vlambda=50):
        # Define and solve the CVXPY problem.
        X = cp.Variable((self.n, self.n))

        diff_up = cp.norm(self.minus_up * X , 1)
        diff_down = cp.norm(self.minus_down * X , 1)
        diff_left = cp.norm(X * self.minus_left, 1)
        diff_right = cp.norm(X * self.minus_right, 1)

        obj = 0.5 * cp.sum_squares(Y - X) + vlambda * (diff_up+diff_down+diff_left+diff_right)

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

    n = 10

    A = np.random.randint(0,20,size=n*n).reshape(n,n)
    print(A)

    tf = Trend_Fiter_2D(n)
    # print(tf.column_shrink)
    tf.test_diff(A)


