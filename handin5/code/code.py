import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time

from functions import *


def lowerBoundLambda(B, eps=0.0001):
    eigvals = np.linalg.eig(B)[0]
    lambda_1 = np.min(eigvals)
    return -np.min(eigvals) + eps


def isPosDef(B):
    return np.min(np.linalg.eig(B)[0]) > 0


def insideRegion(B, g, delta):
    return np.linalg.norm(np.linalg.solve(B, g)) <= delta


def find_p(g, B, delta, max_iter=10, eps=1e-7):
    lambda_l = lowerBoundLambda(B)

    if isPosDef(B) and insideRegion(B, g, delta):
        return np.linalg.solve(B, -g)

    # print("delta:", delta)
    for _ in range(max_iter):
        RTR = B + np.diag(np.repeat(lambda_l, B.shape[0]))
        R = np.linalg.cholesky(RTR)
        # try:
        #     R = np.linalg.cholesky(RTR)
        # except:
        #     # print(np.linalg.norm(q))
        #     exit()
        p = np.linalg.solve(RTR, -g)
        q = np.linalg.solve(R,p)
        # print("pn norm", np.linalg.norm(p))
        lambdaChange = (np.linalg.norm(p) / np.linalg.norm(q)) ** 2 * \
                       ((np.linalg.norm(p) - delta) / delta)

        if lambdaChange < eps:
            break
        lambda_l = lambda_l + lambdaChange
    return p



def updateB(B, y, s):
    temp = (y - B @ s)
    res = B + np.outer(temp, temp) / (temp @ s)
    return res


def sr1_trust_region(f, f_d1, optimum, x, B, max_iter=1000, eta=1e-8, epsilon=1e-7, r=1e-8):
    trust_radius = 1
    opt_dists = np.zeros(max_iter)
    grad_norms = np.zeros(max_iter)

    for i in range(max_iter-1):
        g = f_d1(x)

        grad_norm = np.linalg.norm(g)
        grad_norms[i] = grad_norm
        opt_dists[i] = np.linalg.norm(optimum - x)

        # print("delta:", trust_radius)
        # print("g", np.linalg.norm(g))
        s = find_p(g, B, trust_radius)

        y = f_d1(x + s) - g
        ared = f(x) - f(x + s)
        pred = -(g @ s + 0.5 * s @ B @ s)

        if ared / pred > eta:
            x += s

        if ared / pred > 0.75 and np.linalg.norm(s) > 0.8 * trust_radius:
            trust_radius *= 2
        elif 0.1 > ared / pred or ared / pred > 0.75:
            trust_radius *= 0.5

        temp1 = abs(s @ (y - B @ s))
        temp2 = r * np.linalg.norm(s) * np.linalg.norm(y - np.dot(B, s))

        if temp1 >= temp2 and not np.isclose(y, np.dot(B, s)).all():
            B = updateB(B, y, s)

        # time.sleep(1)
        if trust_radius < epsilon or grad_norm < 1e-10:
        # if grad_norm < epsilon:
            break

    grad_norms[i + 1] = grad_norm
    opt_dists[i + 1] = np.linalg.norm(optimum - x)
    # print(grad_norms[:i+1])

    return x, i + 2, grad_norms, opt_dists


def performanceMessure1(funs, funs_d1, funs_min, B, n_repeats=100, box_size=10):
    n_funs = len(funs)
    accuracy = []
    efficiency = []
    grad_norms = []
    opt_dists = []

    for i, (fun, fun_d1, fun_min) in enumerate(zip(funs, funs_d1, funs_min)):
        print("Running", fun.__name__)
        accuracy_acc = np.zeros(n_repeats)
        efficiency_acc = np.zeros(n_repeats)
        dists_acc = []
        grad_norm_acc = []
        for j in range(n_repeats):
            x = np.random.uniform(-box_size, box_size, 2)
            new_x, iteration, grad_norm, opt_d = sr1_trust_region(fun, fun_d1, fun_min, x, B)
            accuracy_acc[j] = np.linalg.norm(fun_min - new_x)
            efficiency_acc[j] = iteration
            dists_acc.append(np.array(opt_d))
            # print(np.array(grad_norm))
            grad_norm_acc.append(np.array(grad_norm))

        # print(grad_norm_acc)
        # accuracy.append(np.mean(accuracy_acc))
        # efficiency.append(np.mean(efficiency_acc).astype(int))
        # opt_dists.append(np.mean(np.array(dists_acc), axis=0))
        # grad_norms.append(np.mean(np.array(grad_norm_acc), axis=0))
        accuracy.append(np.median(accuracy_acc))
        efficiency.append(np.median(efficiency_acc).astype(int))
        opt_dists.append(np.median(np.array(dists_acc), axis=0))
        grad_norms.append(np.median(np.array(grad_norm_acc), axis=0))

    return np.array(accuracy), np.array(efficiency), np.array(grad_norms), np.array(opt_dists)


def plotgraph(Y, max_ns, fun_labels, y_label, file_name, log=False,
                                                         max_x_factor=1,
                                                         plt_dim=10,
                                                         aspect_ratio=1.3,
                                                         color="green"):
    fig = plt.figure(figsize=(plt_dim, plt_dim * aspect_ratio))
    for i, label in enumerate(fun_labels):
        ax = fig.add_subplot(4, 1, i+1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(label)
        plt.xlabel("number of iterations")

        # max_y = int(max_ns[i] * max_x_factor)
        y = Y[i]
        max_y = y[y>0.0].shape[0]
        if log:
            plt.ylabel(y_label)
            plt.yscale("log")
            # ax.plot(range(1, max_y+1), np.log10(Y[i,:max_y] + 1e-20), color=color)
            ax.plot(range(1, max_y+1), (Y[i,:max_y]), color=color)
        else:
            plt.ylabel(y_label)
            ax.plot(range(1, max_y+1), Y[i,:max_y], color=color)
        # ax.legend()
    plt.subplots_adjust(hspace=0.4)
    # plt.show()
    plt.savefig("../imgs/plt_{}.png".format(file_name), bbox_inches ="tight")
    plt.clf()


def PerformanceMeasure2(d=2, v_min=-3, v_max=4):
    for i,B in enumerate(np.diag([10 ** i] * d) for i in range(v_min, v_max)):
        print(i, B)
        # print("loop:",i)
        # print(SR1_Trust_Region(Rosenbrock_Banana, Rosenbrock_Banana_d1, [10, 12], B)[1])
        # print(SR1_Trust_Region(Log_Ellipsoid, Log_Ellipsoid_d1, [10, 12], B)[1])


def main():
    B = np.diag([1,1])
    # print(sr1_trust_region(Ellipsoid, Ellipsoid_d1, Ellipsoid_min, [10, 12],B))
    # print(sr1_trust_region(Rosenbrock_Banana, Rosenbrock_Banana_d1, [10, 12],B))

    # x_start = np.random.uniform(2,[-10,10])
    # x_start = [0.35684045, 2.33638729]
    # a, b, c, d = sr1_trust_region(Log_Ellipsoid, Log_Ellipsoid_d1, Log_Ellipsoid_min, x_start, B)
    # print(b)

    funs = [Ellipsoid, Rosenbrock_Banana, Log_Ellipsoid, Attractive_Sector5]
    funs_d1 = [Ellipsoid_d1, Rosenbrock_Banana_d1, Log_Ellipsoid_d1, Attractive_Sector5_d1]
    funs_min = [Ellipsoid_min, Rosenbrock_Banana_min, Log_Ellipsoid_min, Attractive_Sector5_min]
    fun_labels=[r"$f_1$",r"$f_2$",r"$f_3$", r"$f_5$"]

    # funs       = [Ellipsoid]
    # funs_d1    = [Ellipsoid_d1]
    # funs_min   = [Ellipsoid_min]
    # fun_labels = [r"$f_1$"]

    # funs       = [Rosenbrock_Banana]
    # funs_d1    = [Rosenbrock_Banana_d1]
    # funs_min   = [Rosenbrock_Banana_min]
    # fun_labels = [r"$f_2$"]

    # funs       = [Log_Ellipsoid]
    # funs_d1    = [Log_Ellipsoid_d1]
    # funs_min   = [Log_Ellipsoid_min]
    # fun_labels = [r"$f_3$"]

    accuracy, efficiency, grad_norms, opt_dists = performanceMessure1(funs, funs_d1, funs_min, B, n_repeats=100)
    print(accuracy)
    print(efficiency)

    plotgraph(opt_dists, efficiency, fun_labels, "dist", "dist", log=True)
    plotgraph(grad_norms, efficiency, fun_labels, "grad norms", "grad_norm", log=True)


if __name__ == '__main__':
    main()
