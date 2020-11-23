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
    return int(sample), int(size)

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
        x = np.zeros((samples, size))
        y = np.zeros((samples, size))
        vx = np.zeros((samples, size))
        vy = np.zeros((samples, size))
        ex = np.zeros((samples, size))
        ey = np.zeros((samples, size))
        ux = np.zeros((size))
        uy = np.zeros((size))
        w = np.zeros((samples))

    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        i = 0
        for row in reader:
            id = int(row['sample'])
            x[id, i] = float(row['x'])
            y[id, i] = float(row['y'])
            vx[id, i] = float(row['x_dot'])
            vy[id, i] = float(row['y_dot'])
            ex[id, i] = float(row['u_x'])
            ey[id, i] = float(row['u_y'])
            if id < 1:
                ux[i] = float(row['u[0]'])
                uy[i] = float(row['u[1]'])
            if id*size + i < samples:
                w[id*size + i] = float(row['w'])
            i += 1
            i %= size
    print(np.mean(x))
    print(np.mean(vx))
    print(np.mean(ex))
    print(np.mean(ux))
    print(np.mean(uy))
    print(np.mean(w))


    plt.figure(0)
    circle1 = plt.Circle((1, 0), 0.01, color='r', fill=False)

    ax = plt.gca()
    ax.cla()
    ax.set_xlim((-1.5, 1.5))
    ax.set_ylim((-1.5, 1.5))
    ax.add_artist(circle1)
    for id, sample in enumerate(x):

        if w[id] > 0.00000000004:
            ax.plot(x[id], y[id])
            ax.quiver(x[id, 1::2], y[id, 1::2], vx[id, 1::2], vy[id, 1::2])
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
