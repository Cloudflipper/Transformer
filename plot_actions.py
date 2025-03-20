import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons
import os
import argparse


def plot_actions(saved_data_dir):

    action_array = np.load(os.path.join(saved_data_dir, "action_data.npy"))



    dt = 0.02
    t = np.arange(0, action_array.shape[0]*dt, dt)
    colors = ["#b81212", "#e06919", "#edf551", "#38f21f", "#07520b", "#10e0d9", "#1028e0", "#7b10e0", "#e01090"]
    # red, orange, yellow, light green, dark green, cyan, blue, purple, pink




    # Actions

    fig, ax = plt.subplots()
    plt.title("actions vs time")
    lines_actions = []
    for action_dim in range(action_array.shape[1]):
        lines_actions.append(ax.plot(t, action_array[:,action_dim], colors[action_dim], label=action_dim))
    plt.subplots_adjust(left = 0.3, bottom = 0.25)
    plt.legend(loc="lower left", bbox_to_anchor=(-0.35, 0.0))
    plt.xlabel("time (s)")
    plt.ylabel("action command")

    # create button
    position_check = plt.axes([0.0, 0.275, 0.08, 0.39])
    # x_pos, y_pos, width, height of button
    activated = [True, True, True, True, True, True, True]
    labels = ["0", "1", "2", "3", "4", "5"]
    check_button_actions = CheckButtons(position_check, labels, activated)
    def select_plot_actions(label):
        index = labels.index(label)

        lines_actions[index][0].set_visible(not lines_actions[index][0].get_visible())
        fig.canvas.draw()

    check_button_actions.on_clicked(select_plot_actions)

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='what is the saved data directory called')
    parser.add_argument('--saved_data', default="saved_data", type=str,
                        help="the directory name that stores the saved data, default name is saved_data")
    
    args = parser.parse_args()
    
    plot_actions(args.saved_data)

   
