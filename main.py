import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D # important even if not used!!!


def rastrigin(X):
    X = np.array(X)
    d = len(X)
    return 10 * d + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=0)


def main():
    print('Hello swarm.')
    x = np.arange(-5, 5, 0.01)
    y = np.arange(-5, 5, 0.01)
    x, y = np.meshgrid(x, y)
    z = rastrigin([x, y])

    # 3d plot projection
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # fig.show()

    plt.contourf(x, y, z)
    plt.show()

    a = 0

if __name__ == '__main__':
    main()