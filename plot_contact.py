import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons
import os
import argparse


def plot_contact(saved_data_dir):
    contact_data = np.load(os.path.join(saved_data_dir, "total_bar_contact_data.npy"))
    print(contact_data.shape)

    dt = 0.02

    def moving_average(vector, ma_steps):
        moving_average_data = np.zeros((vector.shape[0]-ma_steps))
        for i in range(vector.shape[0]-ma_steps):
            moving_average_data[i] = np.mean(vector[i:i+ma_steps])
        return moving_average_data

    t = np.arange(0, contact_data.shape[0]*dt, dt)
    ma_steps = 100 # each data point represents the average of the last 100 data points
    ma_contact_0001_limited = moving_average(contact_data, ma_steps)
    fig = plt.figure()
    plt.title("contact between bars vs time")
    plt.plot(t, contact_data, 'r', label="force")
    plt.plot(t[ma_steps:], ma_contact_0001_limited, 'g', label="moving average: 100")
    plt.ylabel("force (N)")
    plt.xlabel("time (seconds)")

    plt.legend(loc="lower left")

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='what is the saved data directory called')
    parser.add_argument('--saved_data', default="saved_data", type=str,
                        help="the directory name that stores the saved data, default name is saved_data")
    
    args = parser.parse_args()
    
    plot_contact(args.saved_data)