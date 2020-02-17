import numpy as np
import matplotlib.pyplot as plt
from functions import AttractiveSector1, AttractiveSector2, Ellipsoid, Rosenbrock, LogEllipsoid




def main():
    rose = Rosenbrock()
    print(rose.minimize())


if __name__ == "__main__":
    main()
