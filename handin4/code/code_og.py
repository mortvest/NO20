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

# def changeH(H,b=1e-8):
#     val = np.linalg.eig(H)[0]
#     vec = np.linalg.eig(H)[1]

#     for i,v in enumerate(val):
#         if v<=0:
#             val[i]=b

#     res = np.zeros(H.shape)
#     for i in range(val.shape[0]):
#         v = vec[:,i].reshape(2, 1)
#         res += val[i] * (v * v.T)

#     return res

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

def Trust_region(f,f_d1,f_d2,x,max_iter=40,max_step_length=1000,eta=0.2,epsilon=1e-5):
    def m(p, x, f, g,B):
        return f(x) + np.dot(g, p) + 0.5 * (np.dot(np.dot(p, B), p))

    step_length=1
    step_lengths=[step_length]

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
        if distance < epsilon:
            break

    return x,i,step_lengths

def test_max_step():
    xs=range(1,1000,10)
    ys=[]
    for x in xs:
        _,y,_=Trust_region(Log_Ellipsoid, Log_Ellipsoid_d1, Log_Ellipsoid_d2,
                     [44, 33],max_step_length=x)

        ys.append(y)
    return xs,ys


def myplot(x,y,xlabel,ylabel,title):
    plt.title(title)
    plt.plot(x,y,marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def main():
    # print(Trust_region(Ellipsoid, Ellipsoid_d1, Ellipsoid_d2, [44, 33]))
    print(Trust_region(Rosenbrock_Banana,Rosenbrock_Banana_d1,Rosenbrock_Banana_d2,[44,33]))
    #print(Trust_region(Log_Ellipsoid,Log_Ellipsoid_d1,Log_Ellipsoid_d2,[44,33]))

    #op,iter,y=Trust_region(Log_Ellipsoid, Log_Ellipsoid_d1, Log_Ellipsoid_d2, [44, 33])
    #myplot(range(len(y)), y, "Iterations", 'The trust region radius', "Log_Ellipsoid")

    #op, iter, y = Trust_region(Ellipsoid, Ellipsoid_d1, Ellipsoid_d2, [44, 33])
    #myplot(range(len(y)),y,"Iterations",'The trust region radius',"Ellipsoid")


    # xs,ys=test_max_step()
    # myplot(xs, ys, 'Max step length',"Iterations","Log_Ellipsoid")







if __name__ == '__main__':
    main()
