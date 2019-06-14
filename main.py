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

def eggholder(X):
    X = np.array(X)
    y = -(X[1] + 47) * np.sin(np.sqrt(np.abs(X[1] + (X[0] / 2.0) + 47))) - X[0] * np.sin(np.sqrt(np.abs(X[0] - (X[1] + 47))))
    return y

def grav(function, n, d, iteration, low, high, g0=3):
    p = np.random.uniform(low, high, (n, d))
    velocity = np.array([[0 for k in range(d)] for i in range(n)])
    yield p, velocity

    Pbest = p[np.array([function(x) for x in p]).argmin()]
    Gbest = Pbest


    for t in range(iteration):
        csi = np.random.random((n, d))
        eps = np.random.random((1, n))[0]

        fitness = np.array([function(x) for x in p])

        m = np.array([(function(x) - max(fitness)) /
                      (min(fitness) - max(fitness)) for x in p])
        M = np.array([i / sum(m) for i in m])

        G = g0 / np.exp(0.01 * t)
        a = np.array([sum([eps[j] * G * M[j] *
                           (p[j] - p[i]) / (np.linalg.norm(p[i] - p[j]) + 0.001)
                           for j in range(n)]) for i in range(n)])

        v = csi * velocity + np.array([a[i] for i in range(n)])
        p += v
        p = np.clip(p, low, high)
        yield p, velocity

        Pbest = p[np.array([function(x) for x in p]).argmin()]
        if function(Pbest) < function(Gbest):
            Gbest = Pbest

    yield p, velocity

def PSO(func, n, d, iters, low, high, params = {}):
    dp = {'v': 2.4, 'c1': 0.8, 'c2': 0.6, 'm': 0.9, 'decay': 0.2}
    dp.update(params)
    c1, c2 = dp['c1'], dp['c2']
    m = dp['m']
    v_decay = dp['decay']
    p = np.random.uniform(low, high, (n, d))
    v = np.random.uniform(-dp['v'], dp['v'], (n, d))
    # v = np.zeros_like(p)
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
        # v = np.zeros_like(p)
        p = p + v_decay * v
        p = np.clip(p, low, high)
        scores = func(p.T)
        global_best_score_ind = np.argmin(scores)
        global_best_pos = p[global_best_score_ind, :]
        is_curr_better = scores < particle_best_score
        particle_best_score[is_curr_better] = scores[is_curr_better]
        particle_best_pos[is_curr_better] = p[is_curr_better]
        yield p, v

def animate_swarm(swarm, iters, func, rrange):
    fig, ax = plt.subplots()
    x = np.linspace(-rrange*1.1, rrange*1.1, 100)
    y = np.linspace(-rrange*1.1, rrange*1.1, 100)
    x, y = np.meshgrid(x, y)
    z = func([x, y])
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
        ax.set_title(str(frame))
        background = ax.contourf(x, y, z)
        stacked = np.concatenate(pos_history, axis=0)
        history = ax.scatter(stacked[:, 0], stacked[:, 1], color='b', alpha=0.3)
        points = ax.scatter(p[:, 0], p[:, 1], color='r')
        return background, points, history

    ani = FuncAnimation(fig, update, frames=iters, init_func=init, blit=False)
    plt.show()

def plot_hist(func, swarm_hist):
    a = np.array(swarm_hist)[1:, :, :, :]
    bests = np.min(func(a[:, 0, :, :].reshape((-1, 2)).T).reshape((a.shape[0], a.shape[2])), axis=1)
    plt.title('{}'.format(np.min(bests)))
    plt.plot(np.arange(0, len(bests)), bests)

def main():
    print('Hello swarm.')

    use_rast = not True
    use_pso = True

    pop = 10
    num_iter = 500
    func = rastrigin if use_rast else eggholder
    xmin = -5 if use_rast else -512
    xmax = 5 if use_rast else 512
    swarn_gen = PSO if use_pso else grav

    learn_hist = list(swarn_gen(func, pop, 2, num_iter, xmin, xmax))
    plot_hist(func, learn_hist)
    animate_swarm(iter(learn_hist), num_iter, func, xmax)

    a = 0

if __name__ == '__main__':
    main()