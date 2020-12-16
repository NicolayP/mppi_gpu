import matplotlib.pyplot as plt
import numpy as np
import csv

def getMetaDataTraj(csvfile):
    reader = csv.DictReader(csvfile)
    for row in reader:
        x_size = int(row['size_x'])
        u_size = int(row['size_u'])
        break

    return x_size, u_size

def getMetaDataStep(file):
    with open(file) as csvfile:
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

def getBestSampleArg(file):
    samples, size = getMetaDataStep(file)
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        arg_min = 0
        max = -1
        for i, row in enumerate(reader):
            id = int(row['sample'])
            if id*size + i < samples:
                tmp = float(row['w'])
                if tmp > max:
                    max = tmp
                    arg_max = i
    return arg_max, samples, size

def getBestSample(file):
    id, samples, size = getBestSampleArg(file)
    print(id)
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        best_x = np.zeros((size))
        best_y = np.zeros((size))
        i = 0
        for row in reader:
            if int(row['sample']) == id:
                best_x[i] = float(row['x'])
                best_y[i] = float(row['y'])
                i += 1
            if int(row['sample']) > id:
                break
    return best_x, best_y



def plot_files(file):
    x = []
    y = []
    vx = []
    vy = []
    ux = []
    uy = []
    x_size = 0
    u_size = 0
    #my_data = np.genfromtxt(file, delimiter=',')
    #print(my_data.shape)


    with open(file) as csvfile:
        x_size, u_size = getMetaDataTraj(csvfile)
        x = np.zeros((x_size))
        y = np.zeros((x_size))
        vx = np.zeros((x_size))
        vy = np.zeros((x_size))
        ux = np.zeros((u_size))
        uy = np.zeros((u_size))

    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i < u_size :
                ux[i] = float(row['ux'])
                uy[i] = float(row['uy'])
            x[i] = float(row['x'])
            y[i] = float(row['y'])
            vx[i] = float(row['vx'])
            vy[i] = float(row['vy'])


        #ax.yscale('linear')
        #ax.xscale('linear')
        plt.figure(1)
        circle1 = plt.Circle((1, 0), 0.01, color='r', fill=False)

        ax = plt.gca()
        ax.cla()
        ax.set_xlim((-0.5, 1.5))
        ax.set_ylim((-0.5, 0.5))
        ax.add_artist(circle1)
        ax.plot(x, y, '-b')
        '''
        for i in range(u_size):
            ax.quiver(x[:i], y[:i], ux[:i], uy[:i])
            file = "../build/to_plot.csv" + str(i)
            best_x, best_y = getBestSample(file)
            ax.plot(best_x, best_y, '-r')
        '''
    plt.show()



if __name__ == '__main__':
    plot_files("../build/traj_to_plot.csv")
