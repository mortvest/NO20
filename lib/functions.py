from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Case(ABC):
    def apply(self, x):
        return self.apply_vec(np.array([x]))

    @abstractmethod
    def apply_vec(self, _):
        pass

    @abstractmethod
    def derivative(self, _):
        pass

    @abstractmethod
    def derivative_vec(self, _):
        pass

    @abstractmethod
    def hessian(self, _):
        pass

    @abstractmethod
    def hessian_vec(self, _):
        pass

    def __gen_data(self, x_min, x_max, y_min, y_max, n_ticks):
        X = np.linspace(x_min, x_max, n_ticks, True)
        Y = np.linspace(y_min, y_max, n_ticks, True)
        grid = np.transpose([np.tile(X, n_ticks), np.repeat(Y, n_ticks)])
        Z = self.apply(grid).reshape((n_ticks, n_ticks))
        X_space, Y_space = np.meshgrid(X,Y)
        return X_space,Y_space,Z, grid

    def plot_contour(self, x_min, x_max, y_min, y_max, file_name, n_ticks= 1000, imshow=True, levels=None):
        print("Plotting countour for",type(self).__name__)
        X, Y, Z, grid = self.__gen_data(x_min, x_max, y_min, y_max, n_ticks)
        fig, ax = plt.subplots()
        if imshow:
            IM = ax.imshow(Z, extent=(x_min, x_max, y_min, y_max), aspect="auto", origin='lower')
            CS = ax.contour(Z, cmap="Accent", levels=levels, extent=(x_min, x_max, y_min, y_max))
            ax.clabel(CS, inline=1, fontsize=10)
            fig.colorbar(IM)
        else:
            ax.contour(X, Y, Z,cmap="RdBu_r", levels=levels)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig(file_name)
        plt.clf()

    def plot_contour_3d(self, x_min, x_max, y_min, y_max, file_name, n_ticks=1000):
        print("Plotting countour3D for",type(self).__name__)
        X, Y, Z, _ = self.__gen_data(x_min, x_max, y_min, y_max, n_ticks)
        ax = plt.axes(projection="3d")
        ax.contour3D(X, Y, Z, 100, cmap="coolwarm")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z');
        plt.savefig(file_name)
        plt.clf()

    def plot_surface(self, x_min, x_max, y_min, y_max, file_name, n_ticks=1000):
        print("Plotting surface for",type(self).__name__)
        X, Y, Z, _ = self.__gen_data(x_min, x_max, y_min, y_max, n_ticks)
        ax = plt.gca(projection="3d")
        ax.plot_surface(X, Y, Z, cmap="coolwarm")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z');
        plt.savefig(file_name)
        # plt.show()
        plt.clf()

class Ellipsoid(Case):
    def __init__(self, alpha=1000):
        self.alpha = alpha

    def apply_vec(self, x):
        d = x.shape[1]
        i_s = np.arange(0, d)
        exponents = i_s / (d - 1)
        return np.sum(self.alpha**exponents * x**2, axis=1)

    def derivative_vec(self, x):
        d = x.shape[1]
        i_s = np.arange(0, d)
        exponents = i_s / (d - 1)
        return 2 * self.alpha**exponents * x

    def derivative(self, x):
        return self.derivative_vec(np.array([x]))

    def hessian_vec(self, x):
        d = x.shape[1]
        n = x.shape[0]
        i_s = np.arange(0, d)
        exponents = i_s / (d - 1)
        vals = 2 * self.alpha**exponents

        matrix = np.zeros((d,d))
        for i in range(d):
            matrix[i,i] = vals[i]
        return np.repeat(matrix[np.newaxis, :, :], n, axis=0)

    def hessian(self, x):
        return self.hessian_vec(np.array([x]))


class Rosenbrock(Case):
    def apply_vec(self, x):
        if x.shape[1] != 2:
            raise ValueError("x must have two dimensions")
        x1 = x[:,0]
        x2 = x[:,1]
        return (1 - x1)**2 + 100 * (x2 - x1**2)**2

    def derivative_vec(self, x):
        if x.shape[1] != 2:
            raise ValueError("x must have two dimensions")
        x1 = x[:,0]
        x2 = x[:,1]
        dx1 = 2 * (-1 + x1 + 200 * x2**3 - 200 * x1 * x2)
        dx2 = 200 * (x2 - x1**2)
        return np.vstack((dx1, dx2)).T

    def derivative(self, x):
        return self.derivative_vec(np.array([x]))

    def hessian_vec(self, x):
        d = x.shape[1]
        if d != 2:
            raise ValueError("x must have two dimensions")
        x1 = x[:,0]
        x2 = x[:,1]
        n = x.shape[0]

        # initialize 3d array
        result = np.zeros((n, d, d))
        # find second derivatives as 1d arrays
        dx1x1 = 2 + 1200 * x1**2 - 400 * x2
        dx1x2 = -400 * x1
        dx2x2 = 200
        # insert into the 3d array
        result[:,0,0] = dx1x1
        result[:,0,1] = dx1x2
        result[:,1,0] = dx1x2
        result[:,1,1] = dx2x2
        return result

    def hessian(self, x):
        return self.hessian_vec(np.array([x]))


class LogEllipsoid(Ellipsoid):
    def __init__(self, alpha=1000, eps=(10 ** (-16))):
        self.eps = eps
        self.alpha = alpha

    def apply_vec(self, x):
        f1_v = super().apply(x)
        return np.log(self.eps + f1_v)

    def derivative_vec(self, x):
        d = x.shape[1]
        i_s = np.arange(0, d)
        exponents = i_s / (d - 1)
        denom = 1/(self.eps + np.sum(self.alpha**exponents * x**2, axis=1))
        num = 2 * self.alpha**exponents * x
        return denom[:,np.newaxis] * num

    def derivative(self, x):
        return self.derivative_vec(np.array([x]))

    def hessian(self, x):
        d = x.shape[0]
        i_s = np.arange(0, d)
        exponents = i_s / (d - 1)
        j_num = np.tile(self.alpha ** exponents, (d,1)) * x
        i_num = j_num.T
        num = -4 * i_num * j_num
        denom = self.eps + np.sum(self.alpha**exponents * x**2)
        return num/denom

    def hessian_vec(self, _):
        raise NotImplementedError("Not implemented yet")



class AttractiveSector(Case):
    def __init__(self, q=10**8):
        self.q = q

    def q_exp(self, x):
        return np.exp(self.q * x)

    def apply_h(self, x):
        zeros = np.zeros(x.shape)
        return np.log(1 + np.exp(-np.abs(x))) + np.max((x, zeros), axis=0)

    def hessian(self, x):
        d = x.shape[0]
        hess = np.zeros(d * d)
        diag = self.hess_diag(x)
        hess[::d+1] = diag
        return hess.reshape((d,d))

    @abstractmethod
    def hess_diag(self, x):
        pass

    def fun1(self, x):
        return 1/(1 + self.q_exp(x))

    def fun2(self, x):
        return self.q_exp(x)/(1 + self.q_exp(x))


class AttractiveSector1(AttractiveSector):
    def apply_vec(self, x):
        h = self.apply_h(x)
        h_m = self.apply_h(-x)
        return np.sum(h + 100 * h_m, axis=1)

    def derivative(self, x):
        return self.fun1(-x) - 100 * self.fun2(-x)

    def derivative_vec(self, x):
        raise NotImplementedError("Not implemented yet")

    def hess_diag(self, x):
        return (self.fun1(-x) * self.fun2(-x)
            + 100 * (self.q_exp(-x) / (1 + self.q_exp(-x))**2 ))

    def hessian_vec(self, x):
        raise NotImplementedError("Not implemented yet")


class AttractiveSector2(AttractiveSector):
    def apply_vec(self, x):
        h = self.apply_h(x)
        h_m = self.apply_h(-x)
        return np.sum(h**2 + 100 * h_m**2, axis=1)

    def derivative(self, x):
        return 2 * (self.apply_h(x) * self.fun1(-x) - self.apply_h(-x) * self.fun2(-x))

    def derivative_vec(self, x):
        raise NotImplementedError("Not implemented yet")

    def hess_diag(self, x):
        return 2 * self.fun1(-x) * (self.fun2(-x) * self.q * self.apply_h(x) + self.fun1(-x))

    def hessian_vec(self, x):
        raise NotImplementedError("Not implemented yet")


def main():
    att_sec = AttractiveSector2()
    x = att_sec.hessian(np.array([0,0]))
    print(x)

if __name__ == "__main__":
    main()
