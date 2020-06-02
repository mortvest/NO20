import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from methods import newton
import itertools



# def solve_sub_problem(A, b, x, i, foo, bar):
#     # p 56 in the book
#     grad = A @ x + b
#     p_k = np.zeros(b.shape[0])
#     p_k[i] = 1
#     a_k = -(grad @ p_k)/(p_k @ A @ p_k)
#     alpha = a_k * p_k
#     # print("alpha", alpha)
#     return alpha


# def solve_sub_problem(A,b,x,i,a_max, a_min,rho=0.2, c=1e-7, max_iter=5):
#     def fun(x):
#         return x @ A @ x + b @ x
#     # alpha = min(a_max - x[i], 1)
#     alpha = (a_max - x[i])
#     # alpha = 1
#     gradient = A @ x + b
#     p = np.zeros(b.shape[0])
#     p[i] = 1
#     for i in range(max_iter):
#         print("sub iteration", alpha)
#         term1 = fun(x + alpha * p)
#         term2 = fun(x) + c * alpha * np.dot(gradient, p)
#         print("term1", term1, "term2", term2)
#         if term1 <= term2:
#             break
#         alpha*=rho
#     alpha_e = alpha * p
#     return alpha_e


def solve_sub_problem(A,b,x,i,a_max,a_min):
    def fun(alpha):
        v = x + e_i * alpha
        return (1/2) * v @ A @ v + b @ v

    def der(alpha):
        v = x + e_i * alpha
        return np.array(((A @ v + b) @ e_i))

    def con_max(alpha):
        return a_max - (x[i] + alpha)

    def con_min(alpha):
        return (x[i] + alpha) - a_min

    cons = [{'type':'ineq', 'fun': con_max},
            {'type':'ineq', 'fun': con_min}]

    e_i = np.zeros(b.shape[0])
    e_i[i] = 1
    alpha = 1
    # m = minimize(fun, 1, constraints=cons, jac=der, options={"disp": True})
    m = minimize(fun, alpha, constraints=cons, jac=der)
    return m.x * e_i



def coordinate_descent(A, b, mins, maxs, epsilon=1e-5, max_iter=1000):
    if (np.linalg.eigvals(A) < 0).any():
        raise ValueError("Matrix is not positive definite")
    k = 0
    x = mins + ((maxs - mins) / 2)
    if DEBUG:
        print("Starting:",x)
    converged = False

    for it in range(max_iter):
        if DEBUG:
            print("Iteration", it)
        for i in range(b.shape[0]):
            alpha = solve_sub_problem(A, b, x, i, maxs[i], mins[i])
            x += alpha
            g = A @ x + b
            if DEBUG:
                print("x_{}, alpha: {}, x: {}, g: {}".format(i, alpha, x, g))
            if np.linalg.norm(g) < epsilon:
                break
    return x

def exact_1d(A, b, x_v, i, mins, maxs):
    i_given = i
    i_find = 1 - i
    x = x_v[i]
    m = mins[i_find]
    M = maxs[i_find]
    new_xi = -(A[0,1] * x + b[i_find])/(A[i_find, i_find])
    if DEBUG:
        print("new_xi = {}".format(new_xi))
    new_xi = min(max(new_xi, m), M)

    res = np.zeros(2)
    res[i_find] = new_xi
    res[i_given] = x
    return res


def exact_solution(A,b, mins, maxs):
    def fun(x):
        return (1/2) * x @ A @ x + b @ x
    newton_step = -(np.linalg.inv(A) @ b)
    # check if inside the box
    if (maxs - newton_step >= 0).all() and (newton_step - mins >= 0).all():
        if DEBUG:
            print("took step")
        return newton_step
    else:
        if DEBUG:
            print("searching edges")
        best = np.inf
        for i in range(2):
            for v_i, v in enumerate([mins, maxs]):
                if DEBUG:
                    if v_i == 0:
                        print("checking x_{} >= {}".format(i, v[i]))
                    else:
                        print("checking x_{} <= {}".format(i, v[i]))

                coors = exact_1d(A, b, v, i, mins, maxs)
                curr = fun(coors)
                print("curr = {}, coors = {}".format(curr, coors))
                if curr < best:
                    best_coor = coors
                    best = curr
        return best_coor


def main():
    def fun(x):
        return (1/2) * x @ A @ x + b @ x
    A = np.array([[2,1], [1,1]])
    b = np.array([3,4])
    mins = np.array([-50, 0.5])
    # mins = np.array([-20, -20])
    maxs = np.array([-0.5, 1.5])
    # maxs = np.array([-10, -10])

    cd = coordinate_descent(A, b, mins, maxs, max_iter=10)
    es = exact_solution(A, b, mins, maxs)
    diff = np.linalg.norm(cd - es)
    if diff < 1e-4:
        print("SUCCESS")
    else:
        print("FAILED!")
        print("x_cd = {}, x_es = {}, diff = {}".format(cd, es, diff))
        print("f(x_cd) = {}, f(x_es) = {}".format(fun(cd), fun(es)))

    print(exact_1d(A, b, mins, 1, mins, maxs))

if __name__ == "__main__":
    DEBUG = False
    main()


# mat = np.vstack((mins, maxs)).T
# best = np.inf
# for ind, tup in enumerate(list(itertools.product(*mat))):
#     curr = fun(np.array(tup))
#     print(tup, curr)
#     if curr < best:
#         best = curr
#         best_coors = np.array(tup)
# return best_coors
