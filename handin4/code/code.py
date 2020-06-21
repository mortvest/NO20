import traceback

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from functions import *


def lowerBoundLambda(B, eps=0.0001):
    eigvals = np.linalg.eig(B)[0]
    lambda_1 = np.min(eigvals)
    return -np.min(eigvals) + eps

def isPosDef(B):
    return np.min(np.linalg.eig(B)[0]) > 0

def insideRegion(B, g, delta):
    return np.linalg.norm(np.linalg.inv(B) @ g) <= delta

def find_p(g, B, delta, max_iter=10):
    lambda_l = lowerBoundLambda(B)

    if isPosDef(B) and insideRegion(B, g, delta):
        return np.linalg.solve(B, -g)

    for _ in range(max_iter):
        RTR = B + np.diag(np.repeat(lambda_l, B.shape[0]))

        R = np.linalg.cholesky(RTR)

        p = np.linalg.solve(RTR, -g)
        q = np.linalg.solve(R,p)

        lambdaChange = (np.linalg.norm(p) / np.linalg.norm(q)) ** 2 * \
                       ((np.linalg.norm(p) - delta) / delta)

        lambda_l = lambda_l + lambdaChange
    return p


def evaluate_rho(f, m, x, p, g, B):
    return (f(x) - f(x + p)) / (f(x) - m(p, x, f, g, B) + np.finfo(float).eps)


def trust_region(f, f_d1, f_d2, optimum, x, max_iter=1000, max_trust_radius=1000 ,eta=0.2, epsilon=1e-7):
    def m(p, x, f, g,B):
        return f(x) + np.dot(g, p) + 0.5 * (np.dot(np.dot(p, B), p))

    trust_radius = 1
    trust_radia = np.zeros(max_iter)
    opt_dists = np.zeros(max_iter)
    grad_norms = np.zeros(max_iter)

    max_i = 0

    for i in range(max_iter-1):
        # print("i", i)
        trust_radia[i] = trust_radius
        opt_dists[i] = np.linalg.norm(optimum - x)
        # print("curr x", x)

        g = f_d1(x)

        grad_norms[i] = np.linalg.norm(g)

        B = f_d2(x)
        p = find_p(g, B, trust_radius)
        rho = evaluate_rho(f, m, x, p, g, B)

        if rho < 1/4:
            trust_radius *= 1/4
        elif rho > 3/4 and np.linalg.norm(p) == trust_radius:
            trust_radius = min(2 * trust_radius, max_trust_radius)

        # print(p, rho)
        if rho > eta:
            x += p

        distance = np.linalg.norm(p)
        # print(x)
        max_i = i
        if distance < epsilon:
            break


    opt_dists[i+1] = np.linalg.norm(optimum - x)
    # print(opt_dists[:i+1])

    return x, max_i, trust_radia, opt_dists, grad_norms


def performanceMessure(funs, funs_d1, funs_d2, funs_min, n_repeats=100, box_size=10):
    n_funs = len(funs)
    accuracy = []
    efficiency = []
    trust_radia = []
    grad_norms = []
    opt_dists = []

    for i, (fun, fun_d1, fun_d2, fun_min) in enumerate(zip(funs, funs_d1, funs_d2, funs_min)):
        print("Running", fun.__name__)
        accuracy_acc = np.zeros(n_repeats)
        efficiency_acc = np.zeros(n_repeats)
        dists_acc = []
        radia_acc = []
        grad_norms_acc = []
        for j in range(n_repeats):
            x = np.random.uniform(-box_size, box_size, 2)
            new_x, iteration, trust_r, opt_d, grad_n = trust_region(fun, fun_d1, fun_d2, fun_min, x)
            accuracy_acc[j] = np.linalg.norm(fun_min - new_x)
            efficiency_acc[j] = iteration
            dists_acc.append(np.array(opt_d))
            radia_acc.append(np.array(trust_r))
            grad_norms_acc.append(np.array(grad_n))

        # accuracy.append(np.mean(accuracy_acc))
        # efficiency.append(np.mean(efficiency_acc).astype(int))
        # opt_dists.append(np.mean(np.array(dists_acc), axis=0))
        # trust_radia.append(np.mean(np.array(radia_acc), axis=0))
        # grad_norms.append(np.mean(np.array(grad_norms_acc), axis=0))
        accuracy.append(np.median(accuracy_acc))
        efficiency.append(np.median(efficiency_acc).astype(int))
        opt_dists.append(np.median(np.array(dists_acc), axis=0))
        trust_radia.append(np.median(np.array(radia_acc), axis=0))
        grad_norms.append(np.median(np.array(grad_norms_acc), axis=0))

    return np.array(accuracy), np.array(efficiency), np.array(trust_radia), np.array(opt_dists), np.array(grad_norms)



def plotgraph(Y, max_ns, fun_labels, y_label, file_name, log=False,
                                                         max_x_factor=1.5,
                                                         plt_dim=10,
                                                         aspect_ratio=1.3,
                                                         color="green"):
    # fig = plt.figure(figsize=(plt_dim, int(plt_dim * (len(fun_labels)/aspect_ratio))))
    fig = plt.figure(figsize=(plt_dim, plt_dim * aspect_ratio))
    for i, label in enumerate(fun_labels):
        ax = fig.add_subplot(4, 1, i+1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(label)
        plt.xlabel("number of iterations")

        max_y = int(max_ns[i] * max_x_factor) + 1
        Y += 1e-16
        # max_y = max_ns[i] + 1
        # print(Y[0,:max_y])
        if log:
            plt.ylabel(y_label)
            # # ax.set_yscale('symlog', linthreshy=1e-5)
            ax.set_yscale('log')
            ax.plot(range(1, max_y+1), (Y[i,:max_y]), color=color)
        else:
            plt.ylabel(y_label)
            ax.plot(range(1, max_y+1), Y[i,:max_y], color=color)
        # ax.legend()
    plt.subplots_adjust(hspace=0.4)
    plt.show()
    # plt.savefig("../imgs/plt_{}.png".format(file_name), bbox_inches ="tight")
    plt.clf()



def main():
    funs = [Ellipsoid, Rosenbrock_Banana, Log_Ellipsoid, Attractive_Sector5]
    funs_d1 = [Ellipsoid_d1, Rosenbrock_Banana_d1, Log_Ellipsoid_d1, Attractive_Sector5_d1]
    funs_d2 = [Ellipsoid_d2, Rosenbrock_Banana_d2, Log_Ellipsoid_d2, Attractive_Sector5_d2]
    funs_min = [Ellipsoid_min, Rosenbrock_Banana_min, Log_Ellipsoid_min, Attractive_Sector5_min]
    fun_labels=[r"$f_1$",r"$f_2$",r"$f_3$", r"$f_5$"]


    # foo = 1

    # funs = [funs[foo]]
    # funs_d1 = [funs_d1[foo]]
    # funs_d2 = [funs_d2[foo]]
    # funs_min = [funs_min[foo]]
    # fun_labels = [fun_labels[foo]]


    accuracy, efficiency, trust_radia, opt_dists, grad_norms = performanceMessure(funs, funs_d1, funs_d2, funs_min, n_repeats=100)
    print(accuracy)
    print(efficiency)

    # print(opt_dists[0,:efficiency[0] + 3])
    # print(opt_dists[0, efficiency[0]:efficiency[0]+10])

    plotgraph(opt_dists, efficiency, fun_labels, "dist", "dist", log=True)
    # plotgraph(grad_norms, efficiency, fun_labels, "grad_norms", "grad_norms", log=True)



    # foo = 1
    # f = funs[foo]
    # f_d1 = funs_d1[foo]
    # f_d2 = funs_d2[foo]
    # optimum = funs_min[foo]

    # # x_0 = np.array([4.0, 5.0])
    # x_0 = np.random.uniform(-10, 10, 2)
    # x, i, _, _, _ = trust_region(f, f_d1, f_d2, optimum, x_0)
    # print(x)

if __name__ == '__main__':
    main()
