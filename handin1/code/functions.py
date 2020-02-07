from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Case(ABC):
    @abstractmethod
    def apply(self, _):
        pass

    @abstractmethod
    def derivative(self, _):
        pass

    @abstractmethod
    def hessian(self, _):
        pass

    def __gen_data(self, x_min, x_max, y_min, y_max, n_ticks):
        X = np.linspace(x_min, x_max, n_ticks, True)
        Y = np.linspace(y_min, y_max, n_ticks, True)
        grid = np.transpose([np.tile(X, n_ticks), np.repeat(Y, n_ticks)])
        Z = self.apply(grid).reshape((n_ticks, n_ticks))
        X_space, Y_space = np.meshgrid(X,Y)
        return X_space,Y_space,Z, grid

    def plot_contour(self, x_min, x_max, y_min, y_max, n_ticks= 1000, imshow=False):
        X, Y, Z, grid = self.__gen_data(x_min, x_max, y_min, y_max, n_ticks)
        if imshow:
            plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), aspect="auto")
            plt.contour(Z, cmap="Accent", extent=(x_min, x_max, y_min, y_max))
        else:
            # plt.contour(X, Y, Z)
            grad = self.derivative(grid)
            plt.quiver(X,Y, (grad[:,0], grad[:,1]))
        plt.show()

    def plot_contour_3d(self, x_min, x_max, y_min, y_max, n_ticks=1000):
        X, Y, Z, _ = self.__gen_data(x_min, x_max, y_min, y_max, n_ticks)
        ax = plt.axes(projection="3d")
        ax.contour3D(X, Y, Z, 100, cmap="coolwarm")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z');
        plt.show()

    def plot_surface(self, x_min, x_max, y_min, y_max, n_ticks=1000):
        X, Y, Z, _ = self.__gen_data(x_min, x_max, y_min, y_max, n_ticks)
        ax = plt.gca(projection="3d")
        ax.plot_surface(X, Y, Z, cmap="coolwarm")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z');
        plt.show()


class Ellipsoid(Case):
    def __init__(self, alpha=1000):
        self.alpha = alpha

    def apply(self, x):
        d = x.shape[1]
        i_s = np.arange(0, d)
        exponents = i_s / (d - 1)
        return np.sum(self.alpha**exponents * x**2, axis=1)

    def derivative(self, x):
        d = x.shape[1]
        i_s = np.arange(0, d)
        exponents = i_s / (d - 1)
        return 2 * self.alpha**exponents * x

    def hessian(self, x):
        d = x.shape[1]
        n = x.shape[0]
        i_s = np.arange(0, d)
        exponents = i_s / (d - 1)
        vals = 2 * self.alpha**exponents

        matrix = np.zeros((d,d))
        for i in range(d):
            matrix[i,i] = vals[i]
        return np.repeat(matrix[np.newaxis, :, :], n, axis=0)


class Rosenbrock(Case):
    def apply(self, x):
        if x.shape[1] != 2:
            raise ValueError("x must have two dimensions")
        x1 = x[:,0]
        x2 = x[:,1]
        return (1 - x1)**2 + 100 * (x2 - x1**2)**2

    def derivative(self, x):
        if x.shape[1] != 2:
            raise ValueError("x must have two dimensions")
        x1 = x[:,0]
        x2 = x[:,1]
        dx1 = 2 * (-1 + x1 + 200 * x2**3 - 200 * x1 * x2)
        dx2 = 200 * (x2 - x1**2)
        return np.vstack((dx1, dx2)).T

    def hessian(self, x):
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


class LogEllipsoid(Ellipsoid):
    def __init(self, eps=10 ** (-16)):
        self.eps = eps

    def apply(self, x):
        f1_v = super().apply(x)
        return np.log(self.eps + f1_v)

    def derivative(self, x):
        d = x.shape[1]
        i_s = np.arange(0, d)
        exponents = i_s / (d - 1)
        denom = 1/np.sum(self.alpha**exponents * x**2, axis=1)
        num = 2 * self.alpha**exponents * x
        return denom[:,np.newaxis] * num

    def hessian(self, x):
        raise NotImplementedError("Not implemented yet")


class AttractiveSector(Case):
    def apply_h(self, x):
        q = 10 ** 8
        zeros = np.zeros(x.shape)
        return np.log(1 + np.exp(-np.abs(x))) + np.max((x, zeros), axis=0)

class AttractiveSector1(AttractiveSector):
    def apply(self, x):
        h = self.apply_h(x)
        h_m = self.apply_h(-x)
        return np.sum(h + 100 * h_m, axis=1)

    def derivative(self, x):
        raise NotImplementedError("Not implemented yet")

    def hessian(self, x):
        raise NotImplementedError("Not implemented yet")


class AttractiveSector2(AttractiveSector):
    def apply(self, x):
        h = self.__apply_h(x)
        h_m = self.__apply_h(-x)
        return np.sum(h**2 + 100 * h_m**2, axis=1)

    def derivative(self, x):
        raise NotImplementedError("Not implemented yet")

    def hessian(self, x):
        raise NotImplementedError("Not implemented yet")



def main():
    # ell = Ellipsoid()
    # print(ell.hessian(np.array([[1,2], [3,4], [5,6]])))
    # print(ell.derivative(np.array([[1,2], [3,4], [5,6]])))
    # ell.plot_surface(-100, 100, -10, 10)
    rose = Rosenbrock()
    rose.plot_contour(-100, 100, -10, 10)

    # rose.plot_surface(-100, 100, -10, 10)
    # rose.plot_surface(-2, 2, -1, 3)
    # rose.plot_contour(-2, 2, -1, 3)


    # log_ell = LogEllipsoid()
    # print(log_ell.derivative(np.array([[1,2], [3,4], [5,6]])))
    # log_ell.plot_surface(-100, 100, -10, 10, 1000)
    # log_ell.plot_contour(-100, 100, -10, 10, 1000)

    # att_sec1 = AttractiveSector1()
    # att_sec1.plot_contour(-5, 5, -5, 5, imshow=False)
    # att_sec1.plot_surface(-100, 100, -100, 100)
    # att_sec1.plot_contour_3d(-100, 100, -100, 100)

if __name__ == "__main__":
    main()
