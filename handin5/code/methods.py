import numpy as np
from functions import *
import matplotlib.pyplot as plt

def changeH(H,b=1e-8):
    val = np.linalg.eig(H)[0]
    vec = np.linalg.eig(H)[1]

    for i,v in enumerate(val):
        if v<=0:
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


def find_p(g,B,step_length,max_iter=10):
    if isPostiveDef(B):
        p0=-np.dot(np.linalg.inv(B),g)
        if np.linalg.norm(p0)<step_length:
            return p0


    # find lower bound for which lamba makes B PD
    minBound = lowerBoundLambda(B)
    mylambda = minBound + 1e-10
    for _ in range(max_iter):

        RTR=B+np.diag([mylambda]*B.shape[0])


        R=np.linalg.cholesky(RTR)
        p=np.linalg.solve(RTR, -g)
        q=np.linalg.solve(R,p)

        lambdaChange = (np.linalg.norm(p) / np.linalg.norm(q)) ** 2 * \
                       ((np.linalg.norm(p) - step_length) / step_length)

        alpha = 0.99
        n = 0
        while mylambda + alpha**n * lambdaChange < minBound:
            n += 1
        mylambda=mylambda + alpha**n * lambdaChange
    return p



def evaluate_rho(f,m,x,p,g,B):
    return (f(x)-f(x+p))/(f(x)-m(p,x,f,g,B))


def Trust_region(f,f_d1,f_d2,x,max_iter=100,max_step_length=1000,eta=0.2,epsilon=1e-5):
    def m(p, x, f, g,B):
        return f(x) + np.dot(g, p) + 0.5 * (np.dot(np.dot(p, B), p))

    step_length=1
    step_lengths=[step_length]
    distances=[]
    for i in range(max_iter):
        g = f_d1(x)
        B = f_d2(x)
        p=find_p(g,B,step_length)
        rho=evaluate_rho(f,m,x,p,g,B)

        if rho<1/4:
            step_length*=1/4
        else:
            if rho> 3/4 and np.linalg.norm(p)==step_length:
                step_length=min(2*step_length,max_step_length)
            else:
                step_length=step_length

        if rho>eta:
            x+=p
        else:
            x=x

        step_lengths.append(step_length)

        distance = np.linalg.norm(g)
        distances.append(distance)
        if distance < epsilon:
            break

    return x,i,distances


def line_search(fun,x,p,gradient,alpha=1,rho=0.2,c=0.5,max_iter=100):
    for i in range(max_iter):
        term1=fun(x+alpha*p)
        term2=fun(x)+c*alpha*np.dot(gradient,p)
        if term1 <= term2:
            break

        alpha*=rho
    return alpha



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

def newton(fun,fun_d1,fun_d2,x,epsilon=1e-5,max_iter=20000):
    distances=[]
    for i in range(max_iter):
        gradient = fun_d1(x)
        hessian=fun_d2(x)
        hessian=changeH(hessian)
        hessian_inv=np.linalg.inv(hessian)

        direction=-np.dot(hessian_inv,gradient)
        distance=np.linalg.norm(gradient)

        #alpha = line_search(fun, x, direction, gradient)
        alpha=1

        x += alpha * direction


        distances.append(distance)
        if distance < epsilon:
            break

    return x, i,distances