import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

def getMetaData(csvfile):
    reader = csv.DictReader(csvfile)
    sample = 1
    size = 0
    id_prev = 0
    for row in reader:
        if id_prev < int(row['sample']):
            sample += 1
            id_prev = int(row['sample'])
        size += 1
    size /= sample
    return int(sample), int(size)

def plot_file(file, color, step):
    x = []
    y = []
    vx = []
    vy = []
    ex = []
    ey = []
    ux = []
    uy = []
    w = []
    samples = 0
    size = 0

    #my_data = np.genfromtxt(file, delimiter=',')
    #print(my_data.shape)


    with open(file) as csvfile:
        samples, size = getMetaData(csvfile)
        x = np.zeros((samples, size))
        y = np.zeros((samples, size))
        vx = np.zeros((samples, size))
        vy = np.zeros((samples, size))
        ex = np.zeros((samples, size-1))
        ey = np.zeros((samples, size-1))
        ux = np.zeros((size-1))
        uy = np.zeros((size-1))
        ux_prev = np.zeros((size-1))
        uy_prev = np.zeros((size-1))
        w = np.zeros((samples))
        c = np.zeros((samples))
        cost = np.zeros((samples))

        lam = 1

    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        i = 0
        for row in reader:
            id = int(row['sample'])
            x[id, i] = float(row['x'])
            y[id, i] = float(row['y'])
            vx[id, i] = float(row['x_dot'])
            vy[id, i] = float(row['y_dot'])
            if i < size-1 :
                ex[id, i] = float(row['e_x'])
                ey[id, i] = float(row['e_y'])
            if id < 1 and i < size-1:
                ux[i] = float(row['u[0]'])
                uy[i] = float(row['u[1]'])
                ux_prev[i] = float(row['u_prev[0]'])
                uy_prev[i] = float(row['u_prev[1]'])
            if id*size + i < samples:
                w[id*size + i] = float(row['w'])
                c[id*size + i] = float(row['c'])
            i += 1
            i %= size

    for id, sample in enumerate(x):
        for j, element in enumerate(sample):
            if j < size-1:
                cost[id] += (x[id, j+1] - 1.0) * 50. * (x[id, j+1]-1.0) + lam * ux[j] * ex[id, j]
                cost[id] += y[id, j+1] * 50. * y[id, j+1] + lam * uy[j] * ey[id, j]
                cost[id] += vx[id, j+1] * .5 * vx[id, j+1]
                cost[id] += vy[id, j+1] * .5 * vy[id, j+1]
            else:
                cost[id] += (x[id, j]-1.0) * 50. * (x[id, j]-1.0)
                cost[id] += x[id, j] * 50. * x[id, j]
                cost[id] += x[id, j] * .5 * x[id, j]
                cost[id] += x[id, j] * .5 * x[id, j]

    beta = np.min(cost)
    #print(np.argmin(c))
    #print(np.argmin(cost))
    #print(np.min(c))
    #print(beta)
    #print(np.amax(np.abs(cost-c)))
    #print(np.mean(np.abs(cost-c)))
    #print(np.abs(beta-np.min(c)))
    arg = cost-beta
    exp = np.exp(-arg)
    nabla = np.sum(exp)
    weight = exp/nabla
    u_next = np.zeros((size-1, 2))
    u_next[:, 0] = ux_prev
    u_next[:, 1] = uy_prev
    for t in range(size-1):
        for i in range(samples):
            u_next[t, 0] += weight[i]*ex[i, t]
            u_next[t, 1] += weight[i]*ey[i, t]
    #print(u_next[0, :])

    #print(exp)
    #print(nabla)
    #print(weight)

    #print(np.abs(weight-w))
    '''
    df = pd.DataFrame({"c_gpu" : c,
                       "c_python" : cost,
                       "c_diff" : cost - c,
                       "w_gpu" : w,
                       "w_python": weight,
                       "w_diff": weight - w})
    df.to_csv("diff_c.csv", index=False)

    df = pd.DataFrame({"ux_gpu": ux,
                       "ux_python": u_next[:, 0],
                       "ux_diff": ux - u_next[:, 0],
                       "uy_gpu": uy,
                       "uy_python": u_next[:, 1],
                       "uy_diff": uy - u_next[:, 1]})
    df.to_csv("act_next.csv", index=False)'''

    circle1 = plt.Circle((1, 0), 0.01, color='r', fill=False)

    ax = plt.gca()
    #ax.cla()
    ax.set_xlim((-0.2, 1.1))
    ax.set_ylim((-0.2, 0.2))
    ax.add_artist(circle1)
    i = np.argmax(w)
    j = np.argmax(weight)
    #print(i)
    #print(w[i])
    #print(j)
    #print(weight[j])
    #ax.plot(x[j], y[j], '-b')
    ax.plot(x[0], y[0], color, label=step)
    for id, sample in enumerate(x):
        #if w[id] > 0.00001:
        ax.plot(x[id], y[id], color)
        #ax.quiver(x[id, 1::2], y[id, 1::2], vx[id, 1::2], vy[id, 1::2])
        #ax.yscale('linear')
        #ax.xscale('linear')
    print(i)
    print(j)
    ax.plot(x[i], y[i], '-r')


    '''
    for id, sample in enumerate(x):
        plt.figure(id + samples)
        plt.subplot(211)
        plt.plot(range(size), ex[id])
        plt.subplot(212)
        plt.plot(range(size), ey[id])
        '''


if __name__ == '__main__':
    plt.figure(0)
    plot_file("../build/to_plot.csv0", '-b', 't=0')
    plot_file("../build/to_plot.csv250", 'g', 't=250')
    plot_file("../build/to_plot.csv650", '-k', 't=650')
    ax = plt.gca()
    ax.set_title("Sample generation evolution at different timesteps")
    ax.legend()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    #plot_file("../build/to_plot.csv750")

    plt.show()
