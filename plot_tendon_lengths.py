import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.widgets import Button, RadioButtons, CheckButtons
import os

def plot_tendon_lengths(saved_data_dir):


    tendon_length_array = np.load(os.path.join(saved_data_dir, "tendon_data.npy"))


    dt = 0.02
    t = np.arange(0, tendon_length_array.shape[0]*dt, dt)
    colors = ["#b81212", "#e06919", "#edf551", "#38f21f", "#07520b", "#10e0d9", "#1028e0", "#7b10e0", "#e01090"]
    # red, orange, yellow, light green, dark green, cyan, blue, purple, pink






    fig, ax = plt.subplots()
    plt.title("tendon_length vs time")
    lines_tendons = []
    for tendon_num in range(tendon_length_array.shape[1]):
        if tendon_num < 9:
            lines_tendons.append(ax.plot(t, tendon_length_array[:,tendon_num], colors[tendon_num], label=tendon_num))
    plt.subplots_adjust(left = 0.3, bottom = 0.25)
    plt.legend(loc="lower left", bbox_to_anchor=(-0.35, 0.0))
    plt.xlabel("time (s)")
    plt.ylabel("tendon length (m)")


    #create button
    position_check = plt.axes([0.0, 0.275, 0.08, 0.39])
    # x_pos, y_pos, width, height of button
    activated = [True, True, True, True, True, True, True, True, True, True]
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8" ]
    check_button_tendons = CheckButtons(position_check, labels, activated)
    def select_plot_tendons(label):
        index = labels.index(label)

        lines_tendons[index][0].set_visible(not lines_tendons[index][0].get_visible())
        fig.canvas.draw()

    check_button_tendons.on_clicked(select_plot_tendons)


    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='what is the saved data directory called')
    parser.add_argument('--saved_data', default="saved_data", type=str,
                        help="the directory name that stores the saved data, default name is saved_data")
    
    args = parser.parse_args()
    
    plot_tendon_lengths(args.saved_data)