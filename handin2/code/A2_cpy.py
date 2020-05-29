from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt



def Ellipsoid(x,alpha=10,d=2):
    return sum([pow(alpha,((i+1)-1)/(d-1))*(xx**2) for i,xx in enumerate(x)])

def Ellipsoid_d1(x,alpha=10,d=2):
    return [pow(alpha,((i+1)-1)/(d-1))*xx*2 for i,xx in enumerate(x)]



def Rosenbrock_Banana(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def Rosenbrock_Banana_d1(x):
    dfx1=-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2)
    dfx2=200*(x[1]-x[0]**2)
    return [dfx1,dfx2]



def Log_Ellipsoid(x,epsilon=1e-16):
    return np.log(epsilon+Ellipsoid(x))

def Log_Ellipsoid_d1(x,epsilon=1e-16):
    return [Ellipsoid_d1(x)[i]/(Ellipsoid(x)+epsilon) for i in range(len(x))]




def h(x, q=1e3):
    return (np.log(1 + np.exp(-abs(q * x)))+max(0,x)) / q

def Attractive_Sector4(x):
    return sum([h(xx)+100*h(-xx) for xx in x])

def Attractive_Sector4_d1(x,q=1e2):
    return [(np.exp(q*xx)-100)/(np.exp(q*xx)+1) for xx in x ]




def Attractive_Sector5(x):
    return sum([h(xx)**2+100*h(-xx)**2 for xx in x])

def Attractive_Sector5_d1(x,q=1e2):
    return [2*(np.exp(q*xx)*np.log(np.exp(q*xx)+1)-100*np.log(np.exp(q*-xx)+1)) /
            (q*(np.exp(q*xx)+1)) for xx in x ]



def performanceMessure1(method,funs):
    res_iter=np.zeros((10,5))
    res_success = np.zeros((10, 5))
    for i in range(10):
        input = np.random.uniform(-10, 10, 2)

        for j,fun in enumerate(funs):
            m=minimize(fun, input, method=method)
            res_iter[i, j] =m.nit
            res_success[i, j]=m.success


    res_iter=np.mean(res_iter,axis=0)
    res_success = np.mean(res_success, axis=0)
    return res_iter,res_success



def performanceMessure2(method, funs, funs_min):
    res_iter = []
    for i in range(len(funs)):
        temp = np.zeros((10,1000))
        for j in range(10):
            input = np.random.uniform(-10, 10, 2)
            for z in range(1000):
                m=minimize(funs[i], input, method=method,options={"maxiter":1})
                temp[j,z]=distance(m.x, funs_min[i])
                if(m.success or (input[0] == m.x[0] and input[1] == m.x[1] ) ):
                    break
                input = m.x
        res_iter.append(np.mean(temp,axis=0))

    return res_iter



def distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def calploty(fun_d1,input):
    return np.log(np.linalg.norm(fun_d1(input)))


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
    plt.show()


def plotgraph(Y, method, n_iters):
    colors=["pink", "green","red","black","blue"]
    labels=[r"$f_1$",r"$f_2$",r"$f_3$",r"$f_4$",r"$f_5$"]
    for color, y, label, n_iter in zip(colors,Y, labels, n_iters):
        # plt.plot(range(len(y)), y, '-', color=color)
        max_y = int(n_iter * 2)
        plt.plot(range(max_y), y[:max_y], '-', color=color)
        plt.title(label)
        plt.ylabel("Log of distance to the optimum")
        # plt.yscale("log")
        plt.xlabel("Iteration number")
        plt.legend(labels)
        plt.show()


def main():
    funs = [Ellipsoid, Rosenbrock_Banana, Log_Ellipsoid,
            Attractive_Sector4, Attractive_Sector5]

    funs_d1 = [Ellipsoid_d1, Rosenbrock_Banana_d1, Log_Ellipsoid_d1,
               Attractive_Sector4_d1, Attractive_Sector5_d1]

    method1 = "BFGS"

    funs_min=[[0,0],[1,1],[0,0],[0,0],[0,0]]
    ri1,rs1=performanceMessure1(method1, funs)

    plothist(ri1, method1,r"#iterations until $||\nabla\,f|| < 10^{-5}$")

    # res = performanceMessure2(method1, funs, funs_min)

    # plotgraph(res, method1, ri1)


if __name__ == "__main__":
    main()
    '''
    input = np.random.uniform(-10, 10, 2)
    m = minimize(Ellipsoid, input, method="BFGS")
    print(m.success)
    print(m.x)
    print(m.status)
    print(m.message)
    '''

