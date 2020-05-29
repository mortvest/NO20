import traceback
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from functions import *


def line_search(fun,x,p,gradient,alpha=1,rho=0.2,c=1e-5, max_iter=100):
    for i in range(max_iter):
        term1 = fun(x + alpha * p)
        term2 = fun(x)+c*alpha*np.dot(gradient,p)
        if term1 <= term2:
            break

        alpha*=rho
    return alpha


def gradient_descent(fun,fun_d1,fun_d2,x,optimum,epsilon=1e-7,max_iter=20000, ret_fd=False):
    grad_norm_arr = np.zeros(max_iter)
    dist_arr = np.zeros(max_iter)
    f_diffs = np.zeros(max_iter)

    prev_f = fun(x) - fun(optimum)

    for i in range(max_iter):
        gradient=fun_d1(x)
        iteration = i + 1

        grad_norm_sq=np.linalg.norm(gradient) ** 2

        grad_norm = np.linalg.norm(gradient)
        dist = distance(x, optimum)
        grad_norm_arr[i] = grad_norm
        dist_arr[i] = dist

        direction=-gradient

        alpha = line_search(fun, x, direction, gradient, alpha=0.1)

        new_x = x + alpha*direction
        if ret_fd:
            f_diffs[i] = (fun(new_x) - fun(optimum)) / prev_f
        x = new_x

        if grad_norm_sq<epsilon:
            break

    if ret_fd:
        return iteration, f_diffs
    return x, iteration, grad_norm_arr, dist_arr


def changeH(H,b=1e-8):
    val = np.linalg.eig(H)[0]
    vec = np.linalg.eig(H)[1]

    for i,v in enumerate(val):
        if v<0:
            val[i]=b

    res = np.zeros(H.shape)
    for i in range(val.shape[0]):
        v = vec[:,i].reshape(2, 1)
        res += val[i] * (v * v.T)

    return res

def newton(fun,fun_d1,fun_d2,x,optimum,epsilon=1e-7,max_iter=20000):
    grad_norm_arr = np.zeros(max_iter)
    dist_arr = np.zeros(max_iter)
    iteration = 0
    for i in range(max_iter):
        iteration += 1
        gradient = fun_d1(x)
        hessian=fun_d2(x)
        hessian=changeH(hessian)
        hessian_inv=np.linalg.inv(hessian)

        direction=-np.dot(hessian_inv,gradient)
        grad_dot=np.dot(np.dot(gradient,hessian_inv),gradient)

        grad_norm = np.linalg.norm(gradient)
        dist = distance(x, optimum)
        grad_norm_arr[i] = grad_norm
        dist_arr[i] = dist

        alpha = line_search(fun, x, direction, gradient, rho=0.5)
        x += alpha * direction

        # print(x)
        if grad_dot < epsilon:
            break

    return x, iteration, grad_norm_arr, dist_arr

def performanceMessure(funs,funs_d1,funs_d2,funs_min,algos, n_repeats=10, box_size=10):
    n_algos = len(algos)
    n_funs = len(funs)
    accuracy = np.zeros((n_algos, n_funs, n_repeats))
    efficiency = np.zeros((n_algos, n_funs, n_repeats))
    robustness= np.zeros((n_algos, n_funs, n_repeats))
    grad_norm_acc = []
    dist_acc = []

    for i, fun in enumerate(funs):
        print("FUN: {}".format(fun.__name__))
        grad_norm_acc1 = []
        dist_acc1 = []
        for j in range(n_repeats):
            grad_norm_acc2 = []
            dist_acc2 = []
            for z,algo in enumerate(algos):
                x = np.random.uniform(-box_size, box_size, 2)
                newx, iteration, grad_norm_arr, dist_arr = algo(funs[i],
                                                                funs_d1[i],
                                                                funs_d2[i],
                                                                x,
                                                                funs_min[i])
                accuracy[z,i,j] = distance(newx,funs_min[i])
                efficiency[z,i, j] = iteration
                robustness[z,i, j] = 1
                grad_norm_acc2.append(grad_norm_arr)
                dist_acc2.append(dist_arr)

            grad_norm_acc1.append(grad_norm_acc2)
            dist_acc1.append(dist_acc2)
        grad_norm_acc.append(grad_norm_acc1)
        dist_acc.append(dist_acc1)
    accuracy = np.mean(accuracy, axis=2)
    efficiency = (np.mean(efficiency, axis=2)).astype(int)
    robustness = np.mean(robustness, axis=2)
    grad_norm_acc = np.array(grad_norm_acc).mean(axis=2)
    dist_acc = np.array(dist_acc).mean(axis=1)

    return accuracy,efficiency,robustness,grad_norm_acc,dist_acc


def distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


def plotgraph(Y, methods, y_label, plot_name, log, max_x_factor=1):
    labels=[r"$f_1$",r"$f_2$",r"$f_3$",r"$f_4$",r"$f_5$"]
    colors=["blue", "green"]
    fig = plt.figure(figsize=(10,25))
    for i, label in enumerate(labels):
        ax = fig.add_subplot(7, 1, i+1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(label)
        plt.xlabel("number of iterations")
        plt.ylabel(y_label)
        for j, method in enumerate(methods):
            y = Y[i,0]
            max_y = y[y>0].shape[0] * max_x_factor
            if log:
                # ax.plot(range(1, max_y+1), np.log(Y[i,j,:max_y] + 1e-7), color=colors[j], label=method)
                plt.yscale("log")
                ax.plot(range(1, max_y+1), Y[i,j,:max_y], color=colors[j], label=method)
            else:
                ax.plot(range(1, max_y+1), Y[i,j,:max_y], color=colors[j], label=method)
        ax.legend()
    plt.subplots_adjust(hspace=0.4)
    # plt.show()
    plt.savefig("../imgs/plt_{}.png".format(plot_name), bbox_inches ="tight")
    plt.clf()


def performanceMeassure2(fun, fun_d1, fun_min, alphas, d=5, n_tries=100):
    d_fs = []
    bounds = []
    for alpha in alphas:
        f_diffs_acc = []
        max_iter_acc = []
        for i in range(n_tries):
            x_start = np.random.uniform(-10, 10, d)
            fun_p = partial(fun, alpha=alpha, d=d)
            fun_d1_p = partial(fun_d1, alpha=alpha, d=d)
            n_runs, f_diffs = gradient_descent(fun_p, fun_d1_p, None, x_start, fun_min, ret_fd=True)
            f_diffs_acc.append(f_diffs)
            max_iter_acc.append(n_runs)

        max_iter = np.array(max_iter_acc).mean().astype(int)
        f_diffs = np.array(f_diffs_acc).mean(axis=0)
        bound_val = ((alpha-1)/(alpha+1))**2
        bound = np.repeat(bound_val, max_iter)
        d_fs.append(f_diffs[:max_iter])
        bounds.append(bound)
    return d_fs, bounds


def plot_bounds(d_fs, bounds, alphas):
    colors = ["red", "green", "blue", "yellow", "blue"]
    fig = plt.figure(figsize=(10,9))
    # fig = plt.figure()
    for alpha, color, i in zip(alphas, colors, range(len(alphas))):
        ax = fig.add_subplot(len(d_fs), 1, i+1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(r"$\alpha={}$".format(alpha))
        plt.xlabel("number of iterations")
        plt.ylabel("bound")
        xs = np.arange(d_fs[i].shape[0])
        plt.plot(xs, d_fs[i], color=color, label="error norm")
        plt.plot(xs, bounds[i], color=color, linestyle="--", label="bound")
        ax.legend()
    plt.subplots_adjust(hspace=0.45)
    plt.savefig("../imgs/plt_bound.png", bbox_inches ="tight")
    # plt.show()


def main():
    funs = [Ellipsoid, Rosenbrock_Banana, Log_Ellipsoid,
            Attractive_Sector4, Attractive_Sector5]

    funs_d1 = [Ellipsoid_d1, Rosenbrock_Banana_d1, Log_Ellipsoid_d1,
               Attractive_Sector4_d1, Attractive_Sector5_d1]

    funs_d2 = [Ellipsoid_d2, Rosenbrock_Banana_d2, Log_Ellipsoid_d2,
               Attractive_Sector4_d2, Attractive_Sector5_d2]
    funs_min = [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]]

    algos=[gradient_descent,newton]
    methods=["gradient_descent","newton"]

    accuracy, efficiency, robustness, grad_norms, dists =performanceMessure(funs,
                                                                            funs_d1,
                                                                            funs_d2,
                                                                            funs_min,
                                                                            algos,
                                                                            n_repeats=100)
    print(accuracy)
    print(efficiency)
    print(robustness)

    # plothist(accuracy, methods, "dist", "Accuracy")

    plotgraph(grad_norms, methods, r"$||\nabla||$", "grad", log=False)
    # plotgraph(dists, methods, r"$\log($dist$)$", "dist", log=True)
    plotgraph(dists, methods, r"$dist$", "dist", log=True)

    # fun = Ellipsoid
    # fun_d1 = Ellipsoid_d1
    # alphas = np.array([10**i for i in [1,2,3]])

    # d_fs, bounds = performanceMeassure2(fun, fun_d1, np.repeat(0, 5), alphas, n_tries=100)
    # plot_bounds(d_fs, bounds, alphas)


if __name__ == '__main__':
    DEBUG = False
    main()



