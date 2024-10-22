import numpy as np
import matplotlib.pyplot as plt
from functions import *
from scipy.optimize import minimize

# def Ellipsoid(x,alpha=10,d=2):
#     return sum([pow(alpha,((i+1)-1)/(d-1))*(xx**2) for i,xx in enumerate(x)])

# def Ellipsoid_d1(x,alpha=10,d=2):
#     return np.array([pow(alpha,((i+1)-1)/(d-1))*xx*2 for i,xx in enumerate(x)])

# def Ellipsoid_d2(x,alpha=10,d=2):
#     hessian=np.zeros((2,2))
#     hessian[0,0]=pow(alpha,(1-1)/(d-1))*2
#     hessian[0,1]=0
#     hessian[1,0]=0
#     hessian[1,1] = pow(alpha, (2  - 1) / (d - 1)) * 2
#     return hessian



# def Rosenbrock_Banana(x):
#     return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

# def Rosenbrock_Banana_d1(x):
#     dfx1=-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2)
#     dfx2=200*(x[1]-x[0]**2)
#     return np.array([dfx1,dfx2])

# def Rosenbrock_Banana_d2(x):
#     hessian = np.zeros((2, 2))
#     hessian[0,0]=800*x[0]**2-400*(x[1]-x[0]**2)+2
#     hessian[0,1]=-400*x[0]
#     hessian[1,0] = -400 * x[0]
#     hessian[1,1]=200
#     return hessian




# def Log_Ellipsoid(x,epsilon=1e-6):
#     return np.log(epsilon+Ellipsoid(x))

# def Log_Ellipsoid_d1(x,epsilon=1e-6):
#     return np.array([Ellipsoid_d1(x)[i]/(Ellipsoid(x)+epsilon) for i in range(len(x))])

# def Log_Ellipsoid_d2(x,epsilon=1e-6):
#     hessian = np.zeros((2, 2))
#     for i in range(2):
#         for j in range(2):
#             hessian[i, j]=Ellipsoid_d2(x)[i,j]/(Ellipsoid(x)+epsilon) - \
#                           Ellipsoid_d1(x)[i]*Ellipsoid_d1(x)[j]/ (Ellipsoid(x)+epsilon)**2
#     return hessian


# qq=1e7
# def h(x, q=qq):
#     return (np.log(1 + np.exp(-abs(q * x)))) / q+max(0,x)


# def h_d1(x,q=qq):
#     if x>0:
#         return 1/ (1 + np.exp(-q * x))
#     else:
#         return 1-1/(1+np.exp(q*x))

# def h_d2(x,q=qq):
#     if x>0:
#         return np.exp(-q*x)*q/ (1+np.exp(-q*x))**2
#     else:
#         return ((1+np.exp(q*x))/np.exp(q*x)-1)*q/ (1+((1+np.exp(q*x))/np.exp(q*x)-1))**2


# def Attractive_Sector4(x):
#     return sum([h(xx)+100*h(-xx) for xx in x])

# def Attractive_Sector4_d1(x):
#     return np.array([h_d1(xx)-100*h_d1(-xx) for xx in x ])

# def Attractive_Sector4_d2(x):
#     hessian = np.zeros((2, 2))
#     hessian[0, 0] = h_d2(x[0])+100*h_d2(-x[0])
#     hessian[0, 1] = 0
#     hessian[1, 0] = 0
#     hessian[1, 1] = h_d2(x[1])+100*h_d2(-x[1])
#     return hessian




# def Attractive_Sector5(x):
#     return sum([h(xx)**2+100*h(-xx)**2 for xx in x])

# def Attractive_Sector5_d1(x,q=qq):
#     return np.array([2*h_d1(xx)*h(xx)-200*h_d1(-xx)*h(-xx) for xx in x])


# def Attractive_Sector5_d2(x,q=qq):
#     hessian = np.zeros((2, 2))
#     hessian[0, 0] = 2*( h_d2(x[0]) * h(x[0]) + h_d1(x[0]) * h_d1(x[0]) ) - \
#                     200 * ( h_d2(-x[0]) * h(-x[0]) + h_d1(-x[0]) * h_d1(-x[0]) )

#     hessian[0, 1] = 0
#     hessian[1, 0] = 0
#     hessian[1, 1] = 2*( h_d2(x[1]) * h(x[1]) + h_d1(x[1]) * h_d1(x[1]) ) - \
#                     200 * ( h_d2(-x[1]) * h(-x[1]) + h_d1(-x[1]) * h_d1(-x[1]) )
#     return hessian


def minim(case, method="BFGS"):
    input = np.random.uniform(-10, 10, 2)
    # fun = case()
    # m = minimize(fun.apply, input, method=method)
    # m = minimize(fun.apply, input, jac=fun.derivative, method=method)
    m = minimize(Rosenbrock_Banana, input, jac=Rosenbrock_Banana_d1, method=method)
    print(m)
    # print(m.x)
    return m.nit



def main():
    # functions = [Ellipsoid, Rosenbrock, LogEllipsoid, AttractiveSector1, AttractiveSector2]
    # functions = [Rosenbrock]
    minim(None)


if __name__ == "__main__":
    main()
