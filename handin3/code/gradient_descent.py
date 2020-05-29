import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../lib")
from functions import Case, Rosenbrock, Ellipsoid, LogEllipsoid, AttractiveSector1, AttractiveSector2


def steepest_descend(fun):
    assert issubclass(fun.__class__, Case)

def newtons_method(fun):
    assert issubclass(fun.__class__, Case)

def main():
    steepest_descend(Ellipsoid())

if __name__ == "__main__":
    main()
