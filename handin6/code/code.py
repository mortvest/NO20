import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize


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

def solve_sub_problem(A, b, x, g, i, a_max, a_min):
    # p 56 in the book
    a_k = -(g[i] / A[i, i])
    min_ak = a_min - x[i]
    max_ak = a_max - x[i]
    a_chosen = max(min_ak, min(max_ak, a_k))

    # if DEBUG:
    #     print("min_ak = {}, max_ak = {}, a_k = {}, chosen = {}".format(min_ak, max_ak, a_k, a_chosen))

    a_e_i = np.zeros(b.shape[0])
    a_e_i[i] = a_chosen
    return a_e_i, g + a_chosen * A[i]


def kkt_error(x, g, mins, maxs, eps=1e-5):
    h = np.copy(g)
    # zero_filter = np.logical_or(np.logical_and(g < 0, x == mins), np.logical_and(g > 0, x == maxs))
    zero_filter = np.logical_or(np.logical_and(g > 0, (abs(x - mins) <= eps)),
                                np.logical_and(g < 0, (abs(x - maxs) <= eps)))
    h[zero_filter] = 0

    bounded_arr = np.logical_or((abs(x - mins) <= eps), (abs(x - maxs) <= eps))
    n_bounded = bounded_arr[bounded_arr].shape[0]
    # print(bounded_arr, n_bounded)
    return h, n_bounded


def coordinate_descent(A, b, mins, maxs, epsilon=1e-5, max_iter=1000):
    if (np.linalg.eigvals(A) < 0).any():
        raise ValueError("Matrix is not positive definite")
    k = 0
    x = mins + ((maxs - mins) / 2)
    if DEBUG:
        print("Starting:",x)
    h_norms = np.zeros(max_iter)
    n_bounded_vars = np.zeros(max_iter)

    g = A @ x + b
    for it in range(max_iter):
        print(g, x)
        if DEBUG:
            print("Iteration", it)
        for i in range(b.shape[0]):
            alpha, g = solve_sub_problem(A, b, x, g, i, maxs[i], mins[i])
            x += alpha
            h, n_bounded = kkt_error(x, g, mins, maxs)
        h_norm = np.linalg.norm(h)
        h_norms[it] = h_norm
        n_bounded_vars[it] = n_bounded
        if DEBUG:
            print("alpha: {}, x: {}, h: {}, h_norm: {}".format(alpha, x, h, h_norm))
        print(g, x)
        if h_norm < epsilon:
            break
    if DEBUG:
        if it == max_iter - 1:
            print("Stopped at max_iter = {}".format(it))
        else:
            print("Stopped at {}, epsilon reached".format(it))
    return x, h_norms, n_bounded_vars, it


def exact_1d(A, b, x_v, i, mins, maxs):
    i_find = 1 - i
    x = x_v[i]
    m = mins[i_find]
    M = maxs[i_find]
    new_xi = -(A[0,1] * x + b[i_find])/(A[i_find, i_find])
    if DEBUG:
        print("new_xi = {}".format(new_xi))
    new_xi = min(max(new_xi, m), M)

    res = np.zeros(2)
    res[i] = x
    res[i_find] = new_xi
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



def run_test(A, b, mins, maxs, eps=1e-3, max_iter=10000):
    cd, _, _, _= coordinate_descent(A, b, mins, maxs, max_iter=max_iter)
    es = exact_solution(A, b, mins, maxs)
    diff = np.linalg.norm(cd - es)
    is_close = diff < eps
    if is_close:
        print("SUCCESS")
        print("min at x={}".format(cd))
    else:
        print("FAILED!")
        print("x_cd = {}, x_es = {}, diff = {}".format(cd, es, diff))
        print("f(x_cd) = {}, f(x_es) = {}".format(fun(cd), fun(es)))
        raise ValueError("")
    return is_close


def plotgraph(Y, max_ns, fun_labels, y_label, file_name, x_label="number of iterations",
                                                         log=False,
                                                         max_x_factor=1,
                                                         plt_dim=10,
                                                         aspect_ratio=1.3,
                                                         color="green",
                                                         marker=None):
    fig = plt.figure(figsize=(plt_dim, plt_dim * aspect_ratio))
    for i, label in enumerate(fun_labels):
        ax = fig.add_subplot(len(fun_labels), 1, i+1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(label)
        plt.xlabel(x_label)

        # max_y = int(max_ns[i] * max_x_factor)
        y = Y[i]
        max_y = y[y>0.0].shape[0]
        if log:
            plt.ylabel(y_label)
            plt.yscale("log")
            # ax.plot(range(1, max_y+1), np.log10(Y[i,:max_y] + 1e-20), color=color)
            ax.plot(range(1, max_y+1), (Y[i,:max_y]), color=color, marker=marker)
        else:
            plt.ylabel(y_label)
            ax.plot(range(1, max_y+1), Y[i,:max_y], color=color, marker=marker)
        # ax.legend()
    plt.subplots_adjust(hspace=0.4)
    # plt.show()
    plt.savefig("../imgs/plt_{}.png".format(file_name), bbox_inches ="tight")
    plt.clf()



def create_quadratic_problem(d):
    B = np.random.randn(d, d)
    m = -np.ones(d)
    M = np.ones(d)
    b = np.random.normal(0, 0.01, d)
    A = B @ B.T
    return A, b, m, M


def performance_measure1(num_repeats=100, d=100, max_iter=1000):
    hnorm_acc = []
    n_bounded_acc = []
    iter_acc = []
    for i in range(num_repeats):
        print("repeat #{}".format(i+1))
        A, b, m, M = create_quadratic_problem(d)
        _, hnorms, n_bounded, n_iter = coordinate_descent(A, b, m, M, max_iter=max_iter)
        hnorm_acc.append(hnorms)
        iter_acc.append(n_iter)
        n_bounded_acc.append(n_bounded)
    hnorm_avg = np.array(hnorm_acc).mean(axis=0)
    # hnorm_avg = np.median(np.array(hnorm_acc), axis=0)
    n_iter_avg = np.array(iter_acc).mean()
    n_bounded_avg = np.array(n_bounded_acc).mean(axis=0)
    return hnorm_avg, n_bounded_avg, n_iter_avg

def performance_measure2(ds, num_repeats=100):
    iter_arr_acc = []
    for d in ds:
        print("d = {}".format(d))
        if d < 1:
            raise ValueError("d must be greater than zero")
        iter_arr = np.zeros(num_repeats)
        for i in range(num_repeats):
            A, b, m, M = create_quadratic_problem(d)
            _, _, _, n_iter = coordinate_descent(A, b, m, M)
            iter_arr[i] = n_iter
        iter_arr_acc.append(iter_arr)
    n_iter_avg = np.array(iter_arr_acc).mean(axis=1)
    return n_iter_avg

def create_2d_problem():
    beta = 999/1000
    A = np.array([[1, beta], [beta, 1]])
    m = -np.ones(2)
    M = np.ones(2)
    b = np.random.normal(0, 0.01, 2)
    return A, b, m, M



def performance_measure3(num_repeats=100):
    hnorm_acc = []
    iter_acc = []
    for i in range(num_repeats):
        print("repeat #{}".format(i+1))
        A, b, m, M = create_2d_problem()
        _, hnorms, n_bounded, n_iter = coordinate_descent(A, b, m, M)
        hnorm_acc.append(hnorms)
        iter_acc.append(n_iter)
    hnorm_avg = np.array(hnorm_acc).mean(axis=0)
    n_iter_avg = np.array(iter_acc).mean()
    return hnorm_avg, n_iter_avg




def main():
    # print("Testing convergence 2D")
    # hnorms_2d, _ = performance_measure3()
    # hnorms_2d = np.expand_dims(hnorms_2d, axis=0)
    # plotgraph(hnorms_2d, None, [r"2D problem $\beta=0.999$"], r"norm of $h$", "hnorms_2d", aspect_ratio=0.35, log=True)

    # print("Testing convergence multi")
    # hnorms_3d, _, _ = performance_measure1(d=5, max_iter=1000)
    # hnorms_5d, _, _ = performance_measure1(d=10, max_iter=1000)
    # hnorms_10d, _, _ = performance_measure1(d=15, max_iter=1000)
    # hnorms = np.vstack([hnorms_3d, hnorms_5d, hnorms_10d])

    # labels = [r"$d=5$",
    #           r"$d=10$",
    #           r"$d=15$"]
    # plotgraph(hnorms, None, labels, r"norm of $h$", "hnorms", log=True)

    # print("Testing dimensionality")
    # ds = [i for i in range(1,20)]
    # n_iter = performance_measure2(ds, num_repeats=100)
    # n_iter = np.expand_dims(n_iter, axis=0)
    # plotgraph(n_iter, None, ["Dimensionality test"], r"number of iterations", "dim", x_label="d", marker="o", aspect_ratio=0.9)

    A = np.array([[2,1], [1,1]])
    b = np.array([3, 4])
    # b = np.array([0,0])
    mins = np.array([-75, -75])
    maxs = np.array([-50, -50])
    # mins = np.array([5, 5])
    # maxs = np.array([7, 7])
    run_test(A, b, mins, maxs)

    # A, _, mins, maxs = create_quadratic_problem(2)
    # b = np.array([0.0001, 0.00003])

    # run_test(A, b, mins, maxs, max_iter=100)

    # max_iter = 100
    # cd, h_norms, _ = coordinate_descent(A, b, mins, maxs)

    # h_norms = np.expand_dims(h_norms, axis=0)

    # plotgraph(h_norms, None, ["f"], r"norm of $h$", "")



    # A, b, mins, maxs = create_quadratic_problem(10)

    # cd, h_norms, n_bounded, i = coordinate_descent(A, b, mins, maxs, max_iter = 1000)
    # print(h_norms, i)

    # mins = np.repeat(10, 5)
    # maxs = np.repeat(-10,5)
    # x = np.array([9.999, -9.999, 0, 1, 10])
    # g = np.zeros(5)

    # print(kkt_error(x, g, mins, maxs))


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
