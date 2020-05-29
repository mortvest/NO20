from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from functions import *
from functools import partial


def performanceMessure1(method, funs, funs_d1):
    res_iter=np.zeros((10,5))
    res_success = np.zeros((10, 5))
    for i in range(10):
        input = np.random.uniform(-10, 10, 2)

        for j,fun in enumerate(funs):
            m=minimize(fun, input, jac=funs_d1[j], method=method)
            res_iter[i, j] =m.nit
            res_success[i, j]=m.success


    res_iter=np.mean(res_iter,axis=0)
    res_success = np.mean(res_success, axis=0)
    return res_iter,res_success


def performanceMessure2(method, funs, funs_min, funs_d1, low_shelf=1e-3, n_repeats=100):
    def cb_fun(xk, arr, j, f_min):
        nonlocal i_x
        foo = distance(xk, f_min)
        if foo < low_shelf:
            foo = low_shelf
        arr[j,i_x] = foo
        i_x += 1
    res_iter = []
    i_x = 0
    for i in range(len(funs)):
        temp = np.repeat(low_shelf, n_repeats * 1000).reshape((n_repeats,1000))
        for j in range(n_repeats):
            i_x = 0
            input = np.random.uniform(-10, 10, 2)
            m=minimize(funs[i], input, jac=funs_d1[i], method=method, callback=partial(cb_fun,
                                                                       arr=temp,
                                                                       j=j,
                                                                       f_min= funs_min[i]))
        res_iter.append(np.mean(temp,axis=0))
    return res_iter




def distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


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


# def plotgraph(Y, method, n_iters, max_x_factor=2.5):
def plotgraph(Y, method, n_iters, max_x_factor=1):
    labels=[r"$f_1$",r"$f_2$",r"$f_3$",r"$f_4$",r"$f_5$"]
    fig = plt.figure(figsize=(10,17))
    for y, label, n_iter, i in zip(Y, labels, n_iters, range(1, len(labels) + 1)):
        max_y = int(n_iter * max_x_factor)
        ax = fig.add_subplot(7, 1, i)
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.plot(range(max_y), (np.log(y[:max_y])), color="green")
        plt.yscale("log")
        ax.plot(range(max_y), (y[:max_y]), color="green")
        ax.set_title(label)
        plt.xlabel("number of iterations")
        # plt.ylabel(r"$\log($dist$)$")
        plt.ylabel("dist")
        # plt.yscale("log")

    plt.subplots_adjust(hspace=0.7)
    plt.savefig("../imgs/plt.png".format(label), bbox_inches ="tight")
    plt.clf()


def main():
    funs = [Ellipsoid, Rosenbrock_Banana, Log_Ellipsoid,
            Attractive_Sector4, Attractive_Sector5]

    funs_d1 = [Ellipsoid_d1, Rosenbrock_Banana_d1, Log_Ellipsoid_d1,
               Attractive_Sector4_d1, Attractive_Sector5_d1]
    method1 = "BFGS"

    funs_min=[[0,0],[1,1],[0,0],[0,0],[0,0]]
    ri1,rs1=performanceMessure1(method1, funs, funs_d1)

    plothist(ri1, method1,r"#iterations until $||\nabla\,f|| < 10^{-5}$")

    res = performanceMessure2(method1, funs, funs_min, funs_d1)

    plotgraph(res, method1, ri1)


if __name__ == "__main__":
    main()

