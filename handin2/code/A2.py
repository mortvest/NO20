from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from functions import *
from functools import partial


def performanceMessure1(method, funs, funs_d1, funs_min, n_repeats=100):
    n_funs = len(funs)
    res_iter=np.zeros((n_repeats, n_funs))
    res_dist = np.zeros((n_repeats, n_funs))
    # for i in range(n_repeats):
    for i in range(n_repeats):
        input = np.random.uniform(-10, 10, 2)

        for j,fun in enumerate(funs):
            m= minimize(fun, input, jac=funs_d1[j], options={"gtol": 1e-5, "maxiter": 1000}, method=method)
            res_iter[i, j] = m.nit
            res_dist[i, j] = np.linalg.norm(m.x - funs_min[j])

    res_iter = np.mean(res_iter,axis=0)
    res_dist = np.mean(res_dist, axis=0)
    return res_iter, res_dist


def performanceMessure2(method, funs, funs_min, funs_d1, low_shelf=1e-5, n_repeats=100, max_iter=1000):
    def cb_fun(xk, fun_d1, dist_arr, grad_norm_arr, j, f_min):
        nonlocal i_x
        foo = np.linalg.norm(xk - f_min)
        if foo < low_shelf:
            foo = low_shelf
        dist_arr[j,i_x] = foo
        bar = np.linalg.norm(fun_d1(xk))
        grad_norm_arr[j, i_x] = bar
        i_x += 1

    dist_acc = []
    grad_norm_acc = []
    i_x = 0

    for i in range(len(funs)):
        dist_arr = np.repeat(low_shelf, n_repeats * max_iter).reshape((n_repeats, max_iter))
        grad_norm_arr = np.repeat(low_shelf, n_repeats * max_iter).reshape((n_repeats, max_iter))
        for j in range(n_repeats):
            i_x = 0
            input = np.random.uniform(-10, 10, 2)
            fun_d1 = funs_d1[i]
            cb = partial(cb_fun,
                         fun_d1 = fun_d1,
                         dist_arr=dist_arr,
                         grad_norm_arr=grad_norm_arr,
                         j=j,
                         f_min=funs_min[i])
            m =minimize(funs[i], input, jac=funs_d1[i], method=method, options={"gtol": 1e-5, "maxiter":1000}, callback=cb)
        dist_acc.append(np.median(dist_arr, axis=0))
        grad_norm_acc.append(np.median(grad_norm_arr, axis=0))
        # dist_acc.append(np.mean(dist_arr, axis=0))
        # grad_norm_acc.append(np.mean(grad_norm_arr, axis=0))
    return dist_acc, grad_norm_acc


def plothist(y, method_name, ylabel):
    labels=[r"$f_1$",r"$f_2$",r"$f_3$",r"$f_4$",r"$f_5$"]
    x = np.arange(5)

    fig, ax = plt.subplots()
    ax.bar(x, y, color="red")

    ax.set_ylabel(ylabel)
    # ax.set_title('Performance measurement 1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # ax.legend()
    plt.savefig("../imgs/hist.png")
    plt.clf()


def plotgraph(Y, method, n_iters, f_name, y_label, max_x_factor=1):
    labels=[r"$f_1$",r"$f_2$",r"$f_3$",r"$f_4$",r"$f_5$"]
    fig = plt.figure(figsize=(10,25))
    for y, label, n_iter, i in zip(Y, labels, n_iters, range(1, len(labels) + 1)):
        # max_y = int(n_iter * max_x_factor)
        max_y = int(n_iter * max_x_factor) + 1
        ax = fig.add_subplot(7, 1, i)
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.plot(range(max_y), (np.log(y[:max_y])), color="green")
        plt.yscale("log")
        # print(y[:max_y])

        ax.plot(range(max_y), (y[:max_y]), color="green")
        ax.set_title(label)
        plt.xlabel("number of iterations")
        # plt.ylabel(r"$\log($dist$)$")
        plt.ylabel(y_label)
        # plt.yscale("log")

    plt.subplots_adjust(hspace=0.4)
    # plt.show()
    plt.savefig("../imgs/plt_{}.png".format(f_name), bbox_inches ="tight")
    plt.clf()


def main():
    funs = [Ellipsoid, Rosenbrock_Banana, Log_Ellipsoid,
            Attractive_Sector4, Attractive_Sector5]

    funs_d1 = [Ellipsoid_d1, Rosenbrock_Banana_d1, Log_Ellipsoid_d1,
               Attractive_Sector4_d1, Attractive_Sector5_d1]
    # funs = [Log_Ellipsoid]
    # funs_d1 = [Log_Ellipsoid_d1]

    # method1 = "CG"
    method1 = "BFGS"

    funs_min=[[0,0],[1,1],[0,0],[0,0],[0,0]]
    ri1,rs1=performanceMessure1(method1, funs, funs_d1, funs_min)

    print(ri1)
    print(rs1)

    plothist(ri1, method1,r"#iterations until $||\nabla\,f|| < 10^{-5}$")

    dists, grad_norms = performanceMessure2(method1, funs, funs_min, funs_d1)

    plotgraph(dists, method1, ri1, "dists", "dist")
    plotgraph(grad_norms, method1, ri1, "grad_norms", r"$\| \nabla f \|$")


if __name__ == "__main__":
    main()

