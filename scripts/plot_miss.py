import numpy as np
import matplotlib.pyplot as plt
import csv


def plot_miss(filename):
    x_size = 101;
    x_s = np.zeros((x_size))
    x_w = np.zeros((x_size))
    y_s = np.zeros((x_size))
    y_w = np.zeros((x_size))
    v_x_s = np.zeros((x_size))
    v_x_w = np.zeros((x_size))
    v_y_s = np.zeros((x_size))
    v_y_w = np.zeros((x_size))

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            x_s[i] = float(row["x_s"])
            x_w[i] = float(row["x_w"])
            y_s[i] = float(row["y_s"])
            y_w[i] = float(row["y_w"])
            v_x_s[i] = float(row["v_x_s"])
            v_x_w[i] = float(row["v_x_w"])
            v_y_s[i] = float(row["v_y_s"])
            v_y_w[i] = float(row["v_y_w"])

    plt.figure(0)
    ax = plt.gca()
    ax.cla()
    ax.set_title("Position missmatch")
    ax.set_xlim((-1.5, 1.5))
    ax.set_ylim((-1.5, 1.5))
    ax.plot(x_s, y_s, '-b', label="Model")
    ax.plot(x_w, y_w, '-r', label="World")
    ax.legend()

    plt.figure(1)
    ax = plt.gca()
    ax.cla()
    ax.set_title("X Speed missmatch")
    ax.plot(range(x_size), v_x_s, '-b', label="Model")
    ax.plot(range(x_size), v_x_w, '-r', label="World")
    ax.legend()

    plt.figure(2)
    ax = plt.gca()
    ax.cla()
    ax.set_title("Y Speed missmatch")
    ax.plot(range(x_size), v_y_s, '-b', label="Model")
    ax.plot(range(x_size), v_y_w, '-r', label="World")
    ax.legend()

    plt.show()

if __name__ == '__main__':
    plot_miss("../build/missmatch.csv")
