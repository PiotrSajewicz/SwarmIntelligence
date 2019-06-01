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

def PSO(func, n, d, iters, low, high):
    c1, c2 = 0.8, 0.6
    m = 0.9
    v_decay = 0.2
    p = np.random.uniform(low, high, (n, d))
    v = np.random.uniform(-0.6 * 4, 0.6 * 4, (n, d))
    yield p, v
    scores = func(p.T)
    global_best_score_ind = np.argmin(scores)
    global_best_pos = p[global_best_score_ind, :]
    particle_best_pos = p
    particle_best_score = scores
    for i in range(iters):
        r1 = np.random.uniform(0, 1, size=p.shape)
        r2 = np.random.uniform(0, 1, size=1)
        v = m * v + c1 * r1 * -(p-particle_best_pos) + c2 * r2 * -(p-global_best_pos)
        p = p + v_decay * v
        scores = func(p.T)
        global_best_score_ind = np.argmin(scores)
        global_best_pos = p[global_best_score_ind, :]
        is_curr_better = scores < particle_best_score
        particle_best_score[is_curr_better] = scores[is_curr_better]
        particle_best_pos[is_curr_better] = p[is_curr_better]
        yield p, v

def animate_swarm(swarm, iters):
    fig, ax = plt.subplots()
    x = np.arange(-5, 5, 0.01)
    y = np.arange(-5, 5, 0.01)
    x, y = np.meshgrid(x, y)
    z = rastrigin([x, y])
    pos_history = []

    def init():
        background = ax.contourf(x, y, z)
        history = ax.scatter([], [])
        points = ax.scatter([], [])
        return background, history, points

    def update(frame):
        p, v = next(swarm)
        pos_history.append(p)
        ax.clear()
        background = ax.contourf(x, y, z)
        stacked = np.concatenate(pos_history, axis=0)
        history = ax.scatter(stacked[:, 0], stacked[:, 1], color='b', alpha=0.3)
        points = ax.scatter(p[:, 0], p[:, 1], color='r')
        return background, history,  points

    ani = FuncAnimation(fig, update, frames=iters,
                        init_func=init, blit=False)
    plt.show()

def main():
    print('Hello swarm.')

    # 3d plot projection
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # fig.show()

    animate_swarm(PSO(rastrigin, 10, 2, 50, -5, 5), 50)

    a = 0

if __name__ == '__main__':
    main()