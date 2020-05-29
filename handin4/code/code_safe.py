import traceback

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from functions import *


def changeH(H, b=1e-8):
    eig_val = np.linalg.eig(H)[0]
    eig_vec = np.linalg.eig(H)[1]

    eig_val[eig_val <= 0] = b

    res = np.zeros(H.shape)
    for i in range(eig_val.shape[0]):
        v = eig_vec[:,i].reshape(2, 1)
        res += eig_val[i] * (v * v.T)

    return res


def isPostiveDef(H):
    eig = np.linalg.eigh(H)[0]
    return not np.any(eig <= 0)


def lowerBoundLambda(B,mylambda = 0,epsilon=1e-3):
    RTR = B + np.diag([mylambda] * B.shape[0])
    eigh_val = np.linalg.eigh(RTR)[0]
    minval = min(eigh_val)

    return max(-minval, 0)


def find_p(g, B, step_length, max_iter=10, alpha=0.99):
    if isPostiveDef(B):
        p0 = -np.dot(np.linalg.inv(B), g)
        if np.linalg.norm(p0) < step_length:
            return p0

    # find lower bound for which lamba makes B PD
    minBound = lowerBoundLambda(B)
    mylambda = minBound + 1e-10

    for _ in range(max_iter):
        RTR=B+np.diag([mylambda]*B.shape[0])

        try:
            R = np.linalg.cholesky(RTR)
        except:
            print("RTR:", RTR)
            print("B:", B)
            print("mylambda", mylambda)
            traceback.print_exc()
            raise ValueError()

        p = np.linalg.solve(RTR, -g)
        q = np.linalg.solve(R,p)

        lambdaChange = (np.linalg.norm(p) / np.linalg.norm(q)) ** 2 * \
                       ((np.linalg.norm(p) - step_length) / step_length)

        n = 0
        while (mylambda + alpha**n * lambdaChange) < minBound:
            n += 1
        mylambda = mylambda + alpha**n * lambdaChange
    return p


def evaluate_rho(f, m, x, p, g, B):
    zeros = np.zeros(p.shape)
    # return (f(x) - f(x + p)) / (m(zeros, x, f, g, B) - m(p, x, f, g, B) + np.finfo(float).eps)
    return (f(x) - f(x + p)) / f(x) - m(p, x, f, g, B) + np.finfo(float).eps)


# def trust_region(f,f_d1,f_d2,x,max_iter=100,max_step_length=1000,eta=0.2,epsilon=1e-5):
def trust_region(f, f_d1, f_d2, optimum, x, max_iter=1000, max_trust_radius=1000 ,eta=0.2, epsilon=1e-7):
    def m(p, x, f, g,B):
        return f(x) + np.dot(g, p) + 0.5 * (np.dot(np.dot(p, B), p))

    trust_radius = 1
    trust_radia = np.zeros(max_iter)
    opt_dists = np.zeros(max_iter)

    for i in range(max_iter-1):
        trust_radia[i] = trust_radius
        opt_dists[i] = np.linalg.norm(optimum-x)

        g = f_d1(x)
        B = f_d2(x)
        p = find_p(g,B,trust_radius)
        rho = evaluate_rho(f,m,x,p,g,B)

        if rho<1/4:
            trust_radius*=1/4
        else:
            if rho> 3/4 and np.linalg.norm(p) == trust_radius:
                trust_radius = min(2 * trust_radius, max_trust_radius)
            else:
                trust_radius = trust_radius

        if rho > eta:
            x += p
        else:
            x = x

        distance = np.linalg.norm(g)
        if distance < epsilon:
            break

    trust_radia[i+1] = trust_radius
    opt_dists[i+1] = np.linalg.norm(optimum-x)
    return x, i+2, trust_radia, opt_dists


def test_max_step():
    xs = range(1,1000,10)
    ys = []
    for x in xs:
        # print("Running for", x)
        x_final, y, _ = trust_region(Log_Ellipsoid, Log_Ellipsoid_d1, Log_Ellipsoid_d2,
                             [44, 33], max_step_length=x)
        ys.append(y)
        print(x_final)
    return xs, ys


def performanceMessure(funs, funs_d1, funs_d2, funs_min, n_repeats=100, box_size=10):
    n_funs = len(funs)
    accuracy = []
    efficiency = []
    trust_radia = []
    opt_dists = []

    for i, (fun, fun_d1, fun_d2, fun_min) in enumerate(zip(funs, funs_d1, funs_d2, funs_min)):
        print("Running", fun.__name__)
        accuracy_acc = np.zeros(n_repeats)
        efficiency_acc = np.zeros(n_repeats)
        dists_acc = []
        radia_acc = []
        for j in range(n_repeats):
            x = np.random.uniform(-box_size, box_size, 2)
            new_x, iteration, trust_r, opt_d = trust_region(fun, fun_d1, fun_d2, fun_min, x)
            accuracy_acc[j] = np.linalg.norm(fun_min - new_x)
            efficiency_acc[j] = iteration
            dists_acc.append(np.array(opt_d))
            radia_acc.append(np.array(trust_r))

        accuracy.append(np.mean(accuracy_acc))
        efficiency.append(np.mean(efficiency_acc).astype(int))
        opt_dists.append(np.mean(np.array(dists_acc), axis=0))
        trust_radia.append(np.mean(np.array(radia_acc), axis=0))

    return np.array(accuracy), np.array(efficiency), np.array(trust_radia), np.array(opt_dists)


def plotgraph(Y, max_ns, fun_labels, y_label, file_name, log=False, max_x_factor=1.5, plt_w=10, color="green"):
    fig = plt.figure(figsize=(plt_w, int(plt_w * (len(fun_labels)/2))))
    for i, label in enumerate(fun_labels):
        ax = fig.add_subplot(4, 1, i+1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(label)
        plt.xlabel("number of iterations")

        max_y = int(max_ns[i] * max_x_factor)
        if log:
            plt.ylabel("log " + y_label)
            # ax.plot(range(1, max_y+1), np.log(Y[i,j,:max_y] + 1e-7), color=colors[j], label=method)
            ax.plot(range(1, max_y+1), np.log(Y[i,:max_y] + 1e-20), color=color)
        else:
            plt.ylabel(y_label)
            ax.plot(range(1, max_y+1), Y[i,:max_y], color=color)
        # ax.legend()
    plt.subplots_adjust(hspace=0.4)
    # plt.show()
    plt.savefig("../imgs/plt_{}.png".format(file_name), bbox_inches ="tight")
    plt.clf()



def main():
    # print(trust_region(Ellipsoid, Ellipsoid_d1, Ellipsoid_d2, Ellipsoid_min, [44, 33]))
    # print(trust_region(Rosenbrock_Banana,Rosenbrock_Banana_d1,Rosenbrock_Banana_d2, Rosenbrock_Banana_min, [44,33]))

    # op,iter,y=trust_region(Log_Ellipsoid, Log_Ellipsoid_d1, Log_Ellipsoid_d2, [44, 33])
    # myplot(range(len(y)), y, "Iterations", 'The trust region radius', "Log_Ellipsoid")

    # op, iter, y, dists = trust_region(Ellipsoid, Ellipsoid_d1, Ellipsoid_d2, Ellipsoid_min, [44, 33])
    # myplot(range(len(y)),y,"n iterations",'Radius of the thrust region',r"$f_1$")
    # myplot(range(len(y)),dists,"n iterations",'Euclidean distance to optimum',r"$f_1$")


    # xs,ys=test_max_step()
    # myplot(xs, ys, 'Max step length',"Iterations","Log_Ellipsoid")

    # funs = [Ellipsoid, Rosenbrock_Banana, Log_Ellipsoid, Attractive_Sector5]
    # funs_d1 = [Ellipsoid_d1, Rosenbrock_Banana_d1, Log_Ellipsoid_d1, Attractive_Sector5_d1]
    # funs_d2 = [Ellipsoid_d2, Rosenbrock_Banana_d2, Log_Ellipsoid_d2, Attractive_Sector5_d2]
    # funs_min = [Ellipsoid_min, Rosenbrock_Banana_min, Log_Ellipsoid_min, Attractive_Sector5_min]
    # fun_labels=[r"$f_1$",r"$f_2$",r"$f_3$", r"$f_5$"]

    # funs = [Ellipsoid, Rosenbrock_Banana, Attractive_Sector5]
    # funs_d1 = [Ellipsoid_d1, Rosenbrock_Banana_d1, Attractive_Sector5_d1]
    # funs_d2 = [Ellipsoid_d2, Rosenbrock_Banana_d2, Attractive_Sector5_d2]
    # funs_min = [Ellipsoid_min, Rosenbrock_Banana_min, Attractive_Sector5_min]
    # fun_labels=[r"$f_1$",r"$f_2$", r"$f_5$"]

    # funs       = [Log_Ellipsoid]
    # funs_d1    = [Log_Ellipsoid_d1]
    # funs_d2    = [Log_Ellipsoid_d2]
    # funs_min   = [Log_Ellipsoid_min]
    # fun_labels = [r"$f_3$"]


    # accuracy, efficiency, trust_radia, opt_dists = performanceMessure(funs, funs_d1, funs_d2, funs_min)
    # print(accuracy)
    # print(efficiency)

    # plotgraph(opt_dists, efficiency, fun_labels, "dist", "dist")
    # plotgraph(opt_dists, efficiency, fun_labels, "trust radius", "radius")
    print(trust_region(Log_Ellipsoid,Log_Ellipsoid_d1,Log_Ellipsoid_d2, Log_Ellipsoid_min, [10,10]))

if __name__ == '__main__':
    main()
