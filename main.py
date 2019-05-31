import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D # important even if not used!!!
from matplotlib.animation import FuncAnimation


def rastrigin(X):
    X = np.array(X)
    d = len(X)
    return 10 * d + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=0)

def PSO(n, d, iters, low, high):
    p = np.random.uniform(low, high, (n, d))
    v = np.random.uniform(-0.6, 0.6, (n, d)) * 4
    yield p, v
    for i in range(iters):
        p = p + 0.1 * v
        v = 0.95 * v
        yield p, v

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


    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'ro')

    def init():
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(-1, 1)
        return ln,

    def update(frame):
        xdata.append(frame)
        ydata.append(np.sin(frame))
        ln.set_data(xdata, ydata)
        return ln,

    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                        init_func=init, blit=True)
    plt.show()

    # for p, v in None:
    #     f = plt.figure()
    #     ax = plt.gca()
    #     ax.contourf(x, y, z)
    #     ax.scatter(p[:, 0], p[:, 1], color='r')
    #     f.show()

    a = 0

if __name__ == '__main__':
    main()