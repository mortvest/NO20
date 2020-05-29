import numpy as np
from functions import *
import matplotlib.pyplot as plt


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
    minval=min(eigh_val)

    return max(-minval, 0)




def find_p(g, B, step_length, max_iter=10, alpha=0.99):
    if isPostiveDef(B):
        p0 = -np.dot(np.linalg.inv(B), g)
        if np.linalg.norm(p0) < step_length:
            return p0

    # find lower bound for which lamba makes B PD
    minBound = lowerBoundLambda(B)
    mylambda = minBound + 1e-10

    # print("Lower bound: ", minBound)
    # print("My Lambda: ", mylambda)
    for _ in range(max_iter):
        # print("B:", B)
        RTR=B+np.diag([mylambda]*B.shape[0])
        # print("RTB:", RTR)


        R = np.linalg.cholesky(RTR)
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
    return (f(x) - f(x + p)) / (f(x) - m(p, x, f, g, B))


def trust_region(f,f_d1,f_d2,x,max_iter=40,max_step_length=1000,eta=0.2,epsilon=1e-5):
# def trust_region(f,f_d1,f_d2,x,max_iter=100,max_step_length=1000,eta=0.2,epsilon=1e-5):
    def m(p, x, f, g,B):
        return f(x) + np.dot(g, p) + 0.5 * (np.dot(np.dot(p, B), p))

    step_length = 1
    step_lengths = [step_length]

    for i in range(max_iter):
        g = f_d1(x)
        B = f_d2(x)
        p = find_p(g,B,step_length)
        rho = evaluate_rho(f,m,x,p,g,B)

        if rho<1/4:
            step_length*=1/4
        else:
            if rho> 3/4 and np.linalg.norm(p) == step_length:
                step_length = min(2 * step_length, max_step_length)
            else:
                step_length = step_length

        if rho > eta:
            x += p
        else:
            x = x

        step_lengths.append(step_length)

        distance = np.linalg.norm(g)
        if distance < epsilon:
            print("Distance condition met")
            break

    return x, i, step_lengths

# def test_max_step():
#     xs = range(1,1000,10)
#     ys = []
#     for x in xs:
#         # print("Running for", x)
#         x_final, y, _ = trust_region(Log_Ellipsoid, Log_Ellipsoid_d1, Log_Ellipsoid_d2,
#                              [44, 33], max_step_length=x)
#         ys.append(y)
#     return xs, ys


def myplot(x,y,xlabel,ylabel,title):
    plt.title(title)
    plt.plot(x,y,marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def main():
    print(trust_region(Ellipsoid, Ellipsoid_d1, Ellipsoid_d2, [44, 33]))
    # print(trust_region(Rosenbrock_Banana,Rosenbrock_Banana_d1,Rosenbrock_Banana_d2,[44,33]))
    # print(trust_region(Log_Ellipsoid,Log_Ellipsoid_d1,Log_Ellipsoid_d2,[44,33]))

    # op,iter,y=trust_region(Log_Ellipsoid, Log_Ellipsoid_d1, Log_Ellipsoid_d2, [44, 33])
    # myplot(range(len(y)), y, "Iterations", 'The trust region radius', "Log_Ellipsoid")

    # op, iter, y = trust_region(Ellipsoid, Ellipsoid_d1, Ellipsoid_d2, [44, 33])
    # myplot(range(len(y)),y,"Iterations",'The trust region radius',"Ellipsoid")


    # xs,ys=test_max_step()
    # myplot(xs, ys, 'Max step length',"Iterations","Log_Ellipsoid")


if __name__ == '__main__':
    main()
