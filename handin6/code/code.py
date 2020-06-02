import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from methods import newton
import itertools


def solve_sub_problem(A, b, x, i, a_max, a_min):
    # p 56 in the book
    grad = A @ x + b
    p_k = np.zeros(b.shape[0])
    p_k[i] = 1
    # a_k = -(grad @ p_k) / (p_k @ A @ p_k)
    a_k = -(grad[i] / A[i, i])

    min_ak = a_min - x[i]
    max_ak = a_max - x[i]
    a_chosen = max(min_ak, min(max_ak, a_k))

    # if DEBUG:
    #     print("min_ak = {}, max_ak = {}, a_k = {}, chosen = {}".format(min_ak, max_ak, a_k, a_chosen))

    alpha = a_chosen * p_k

    return alpha


# def solve_sub_problem(A,b,x,i,a_max,a_min):
#     # solve using scipy.minimize
#     def fun(alpha):
#         v = x + e_i * alpha
#         return (1/2) * v @ A @ v + b @ v

#     def der(alpha):
#         v = x + e_i * alpha
#         return np.array(((A @ v + b) @ e_i))

#     def con_max(alpha):
#         return a_max - (x[i] + alpha)

#     def con_min(alpha):
#         return (x[i] + alpha) - a_min

#     cons = [{'type':'ineq', 'fun': con_max},
#             {'type':'ineq', 'fun': con_min}]

#     e_i = np.zeros(b.shape[0])
#     e_i[i] = 1
#     alpha = 1
#     # m = minimize(fun, 1, constraints=cons, jac=der, options={"disp": True})
#     m = minimize(fun, alpha, constraints=cons, jac=der)
#     return m.x * e_i

def stopping_criterion(A, x, b, mins, maxs):
    g = A @ x + b
    h = np.copy(g)
    zero_filter = np.logical_or(np.logical_and(g < 0, x == mins), np.logical_and(g > 0, x == maxs))
    h[zero_filter] = 0
    return h


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
            h = stopping_criterion(A, x, b, mins, maxs)

        h_norm = np.linalg.norm(h)
        if DEBUG:
            print("alpha: {}, x: {}, h: {}, h_norm: {}".format(alpha, x, h, h_norm))
        if h_norm < epsilon:
            break
    print("Stopped at", it)
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
                # if DEBUG:
                #     if v_i == 0:
                #         print("checking x_{} >= {}".format(i, v[i]))
                #     else:
                #         print("checking x_{} <= {}".format(i, v[i]))

                coors = exact_1d(A, b, v, i, mins, maxs)
                curr = fun(coors)
                # if DEBUG:
                #     print("curr = {}, coors = {}".format(curr, coors))
                if curr < best:
                    best_coor = coors
                    best = curr
        return best_coor


def main():
    def fun(x):
        return (1/2) * x @ A @ x + b @ x
    A = np.array([[2,1], [1,1]])
    b = np.array([3,4])
    mins = np.array([-75, -75])
    # mins = np.array([-20, -20])
    maxs = np.array([75, 75])
    # maxs = np.array([-10, -10])

    cd = coordinate_descent(A, b, mins, maxs, max_iter=1000)
    es = exact_solution(A, b, mins, maxs)
    diff = np.linalg.norm(cd - es)
    if diff < 1e-3:
        print("SUCCESS")
        print("min at x={}".format(cd))
    else:
        print("FAILED!")
        print("x_cd = {}, x_es = {}, diff = {}".format(cd, es, diff))
        print("f(x_cd) = {}, f(x_es) = {}".format(fun(cd), fun(es)))


if __name__ == "__main__":
    DEBUG = True
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
