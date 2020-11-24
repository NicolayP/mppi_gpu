import matplotlib.pyplot as plt
import numpy as np
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
    return int(sample), int(size-1)

def plot_file(file):
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
        x = np.zeros((samples, size+1))
        y = np.zeros((samples, size+1))
        vx = np.zeros((samples, size+1))
        vy = np.zeros((samples, size+1))
        ex = np.zeros((samples, size))
        ey = np.zeros((samples, size))
        ux = np.zeros((size))
        uy = np.zeros((size))
        ux_prev = np.zeros((size))
        uy_prev = np.zeros((size))
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
            if i < size :
                ex[id, i] = float(row['e_x'])
                ey[id, i] = float(row['e_y'])
            if id < 1 and i < size:
                ux[i] = float(row['u[0]'])
                uy[i] = float(row['u[1]'])
                ux_prev[i] = float(row['u_prev[0]'])
                uy_prev[i] = float(row['u_prev[1]'])
            if id*size + i < samples:
                w[id*size + i] = float(row['w'])
                c[id*size + i] = float(row['c'])
            i += 1
            i %= (size + 1)

    for id, sample in enumerate(x):
        for j, element in enumerate(sample):
            if j < size:
                cost[id] += (x[id, j+1] - 1.0) * 50. * (x[id, j+1]-1.0) + lam * ux[j] * ex[id, j]
                cost[id] += y[id, j+1] * 50. * y[id, j+1] + lam * uy[j] * ey[id, j]
                cost[id] += vx[id, j+1] * .25 * vx[id, j+1]
                cost[id] += vy[id, j+1] * .25 * vy[id, j+1]
            else:
                cost[id] += (x[id, j]-1.0) * 50. * (x[id, j]-1.0)
                cost[id] += x[id, j] * 50. * x[id, j]
                cost[id] += x[id, j] * .25 * x[id, j]
                cost[id] += x[id, j] * .25 * x[id, j]

    beta = np.min(cost)
    print(np.max(cost))
    print(beta)
    print(np.abs(cost-c))
    print(np.mean(np.abs(cost-c)))
    print(np.abs(beta-np.min(c)))
    arg = cost-beta
    exp = np.exp(-arg)
    nabla = np.sum(exp)
    weight = exp/nabla
    u_next = np.zeros((size, 2))
    for t in range(size):
        for i in range(samples):
            u_next[t, 0] += weight[i]*ex[i, t]
            u_next[t, 1] += weight[i]*ey[i, t]
    print(u_next[0, :])

    #print(exp)
    #print(nabla)
    #print(weight)

    print(np.abs(weight-w))


    plt.figure(0)
    circle1 = plt.Circle((1, 0), 0.01, color='r', fill=False)

    ax = plt.gca()
    ax.cla()
    ax.set_xlim((-1.5, 1.5))
    ax.set_ylim((-1.5, 1.5))
    ax.add_artist(circle1)
    for id, sample in enumerate(x):
        if cost[id] < 2250 :
            ax.plot(x[id], y[id], '-b')
            print("Here: ")
            print(id)
            print(weight[id])
        if w[id] > 0.01:
            ax.plot(x[id], y[id], '-r')
            print("This: ")
            print(id)
            print(weight[id])
        #ax.quiver(x[id, 1::2], y[id, 1::2], vx[id, 1::2], vy[id, 1::2])
        #ax.yscale('linear')
        #ax.xscale('linear')


    '''
    for id, sample in enumerate(x):
        plt.figure(id + samples)
        plt.subplot(211)
        plt.plot(range(size), ex[id])
        plt.subplot(212)
        plt.plot(range(size), ey[id])
        '''
    plt.show()


if __name__ == '__main__':
    plot_file("../build/to_plot.csv")
