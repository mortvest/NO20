import numpy as np
from functions import *
from methods import *
import time

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

def isPostiveDef(H):
    val = np.linalg.eigh(H)[0]
    for v in val:
        if v<=0:
            return False
    return True

def lowerBoundLambda(B,mylambda = 0,epsilon=1e-3):
    RTR = B + np.diag([mylambda] * B.shape[0])
    eigh_val = np.linalg.eigh(RTR)[0]
    minval=min(eigh_val)

    return max(-minval, 0)

def solve_subproblem(g,B,step_length,mylambda=0.1,max_iter=10):
    if isPostiveDef(B):
        p0 = -np.dot(np.linalg.inv(B), g)
        if np.linalg.norm(p0) < step_length:
            return p0

    # find lower bound for which lamba makes B PD
    minBound = lowerBoundLambda(B)
    mylambda = minBound + 1e-7
    for _ in range(max_iter):

        RTR = B + np.diag([mylambda] * B.shape[0])

        R = np.linalg.cholesky(RTR)
        p = np.linalg.solve(RTR, -g)
        q = np.linalg.solve(R, p)

        lambdaChange = (np.linalg.norm(p) / np.linalg.norm(q)) ** 2 * \
                       ((np.linalg.norm(p) - step_length) / step_length)

        alpha = 0.5
        n = 0
        while mylambda + alpha ** n * lambdaChange < minBound:
            n += 1
        mylambda = mylambda + alpha ** n * lambdaChange
    return p

def updateB(B,y,s):
    temp=(y-np.dot(B,s))
    return B+np.outer(temp,temp)/np.dot(temp,s)


def SR1_Trust_Region(f,f_d1,x,B,max_iter=1000,eta=0,epsilon=1e-5,r=1e-8):
    step_length=1
    distances=[]
    for i in range(max_iter):
        g=f_d1(x)
        s=solve_subproblem(g, B, step_length)

        y=f_d1(x+s)-g
        ared=f(x)-f(x+s)
        pred=-(np.dot(g, s)+0.5*np.dot(np.dot(s, B), s))

        if ared/pred >eta:
            x+=s

        if ared/pred >0.75 and np.linalg.norm(s)>0.8*step_length:
            step_length*=2

        elif 0.1 > ared/pred or ared/pred > 0.75:
            step_length*=0.5

        temp1=abs(np.dot(s,y-np.dot(B,s)))
        temp2=r*np.linalg.norm(s)*np.linalg.norm(y-np.dot(B,s))
        if temp1>=temp2:
            B=updateB(B,y,s)



        distance = np.linalg.norm(g)
        distances.append(distance)
        if distance < epsilon:
            break

    return x,i,distances


def PerformanceMeasurement1():
    fun1=[Rosenbrock_Banana,Rosenbrock_Banana_d1,Rosenbrock_Banana_d2]
    fun2 = [Log_Ellipsoid, Log_Ellipsoid_d1, Log_Ellipsoid_d2]
    x = [10, 12]
    res=[]
    for fun in [fun1,fun2]:
        _,i1,d1=newton(fun[0],fun[1],fun[2],x)
        _,i2,d2=Trust_region(fun[0],fun[1],fun[2],x)
        _,i3,d3=SR1_Trust_Region(fun[0],fun[1],x,np.diag([1,1]))
        res.append([d1,d2,d3])

    plotgraph(res[0], "Rosenbrock_Banana")
    plotgraph(res[1], "Log_Ellipsoid")
    plt.show()

def PerformanceMeasurement2():
    B0 = np.diag([0.01, 0.01])
    B1 = np.diag([0.1, 0.1])
    B2 = np.diag([1, 1])
    B3 = np.diag([10, 10])
    B4 = np.diag([100, 100])
    B5 = np.diag([1000, 1000])

    for i,B in enumerate([B0,B1,B2,B3,B4,B5]):
        print("loop:",i)
        print(SR1_Trust_Region(Rosenbrock_Banana, Rosenbrock_Banana_d1, [10, 12], B)[1])
        print(SR1_Trust_Region(Log_Ellipsoid, Log_Ellipsoid_d1, [10, 12], B)[1])


def plotgraph(Y,title):
    colors=["red","blue","yellow"]
    for color,y in zip(colors,Y):
        plt.semilogy(range(len(y)), y, '-', color=color)
    plt.title(title)
    plt.ylabel("Norm of gradient")
    plt.xlabel("Iteration")
    plt.legend(["newton","Trust_region","SR1_Trust_Region"])
    plt.show()


def main():
    B=np.diag([1,1])
    # print(SR1_Trust_Region(Rosenbrock_Banana, Rosenbrock_Banana_d1, [10, 12],B))
    print(SR1_Trust_Region(Log_Ellipsoid, Log_Ellipsoid_d1, [10, 12],B))

    #PerformanceMeasurement1()
    # PerformanceMeasurement2()

if __name__ == '__main__':
    main()
