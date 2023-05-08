# This file is use to plot the training data result

import matplotlib.pyplot as plt
import csv
import numpy as np

# Filepath that store the training data for transfer learning model
# Please modify this path to the path that store the csv data file generated
# filename1 = "exp/trial_1/BipedalWalker-v3-tl-hardcore-data.csv"
filename1 = "BipedalWalker-v3-tl-hardcore-data.csv"

# Read the dataset stored for Transfer learning model
with open(filename1, "r", newline="") as csvFile:
    reader = csv.reader(csvFile, delimiter=",")
    episodes_1 = []
    ep_rs_1 = []
    avg_scores_1 = []
    EPMs_1 = []
    TTMs_1 = []
    avg_rmse_q_loss_1 = []
    steps_1 = []
    for row in reader:
        episode_1 = int(row[0])
        ep_r_1 = float(row[1])
        avg_score_1 = float(row[2])
        EPM_1 = int(row[3])
        TTM_1 = int(row[4])
        avg_rmse_1 = float(row[5])
        step_1 = int(row[6])

        episodes_1.append(episode_1)
        ep_rs_1.append(ep_r_1)
        avg_scores_1.append(avg_score_1)
        EPMs_1.append(EPM_1)
        TTMs_1.append(TTM_1)
        avg_rmse_q_loss_1.append(avg_rmse_1)
        steps_1.append(step_1)
    
# filename2 = "exp/trial_1/BipedalWalker-v3-hardcore-scratch-data.csv"
filename2 = "BipedalWalker-v3-hardcore-scratch-data.csv"

# Read the dataset stored for scratch model
with open(filename2, "r", newline="") as csvFile:
    reader = csv.reader(csvFile, delimiter=",")
    episodes_2 = []
    ep_rs_2 = []
    avg_scores_2 = []
    EPMs_2 = []
    TTMs_2 = []
    avg_rmse_q_loss_2 = []
    steps_2 = []
    for row in reader:
        episode_2 = int(row[0])
        ep_r_2 = float(row[1])
        avg_score_2 = float(row[2])
        EPM_2 = int(row[3])
        TTM_2 = int(row[4])
        avg_rmse_2 = float(row[5])
        step_2 = int(row[6])
        episodes_2.append(episode_2)
        ep_rs_2.append(ep_r_2)
        avg_scores_2.append(avg_score_2)
        EPMs_2.append(EPM_2)
        TTMs_2.append(TTM_2)
        avg_rmse_q_loss_2.append(avg_rmse_2)
        steps_2.append(step_2)

avg_x_1 = [np.mean(ep_rs_1[np.max([0, i - 100]):i]) for i in range(len(ep_rs_1))]
avg_x_2 = [np.mean(ep_rs_2[np.max([0, i - 100]):i]) for i in range(len(ep_rs_2))]

# Find the highest score from the dataset in different agents
def best_score():
    print("Maximum Scores in TL TD3: " + str(max(ep_rs_1)))
    print("Maximum Scores in Scratch TD3: " + str(max(ep_rs_2)))


# function to plot the average reward for both experiment for transfer learning model and scratch model
def plot_avg_reward():
    plt.figure(dpi=200)
    plt.title('BipedalWalker-Hardcore with Small Experience Data')
    plt.plot(range(len(avg_x_1)), avg_x_1, label="TL TD3")
    plt.plot(range(len(avg_x_2)), avg_x_2, label="Scratch TD3")
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.ylim(-300, 300)
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("Compariosn average reward - small experience resource")
    plt.show() 

# plot the rmse of critic loss for experiment 1 (large experience training data)
def plot_avg_rmse_1():
    plt.plot(np.arange(1, len(avg_rmse_q_loss_1) + 1), avg_rmse_q_loss_1, label='TL TD3')
    plt.plot(np.arange(1, len(avg_rmse_q_loss_2) + 1), avg_rmse_q_loss_2, label='Scratch TD3')
    plt.ylabel('Average RMSE Crtic Loss')
    plt.xlabel('Episode #')
    plt.legend(loc="lower right")
    plt.ylim(0, 3)
    plt.grid()
    plt.title("Average RMSE of critic loss with Large Experience Data")
    plt.savefig("Average RMSE of critic loss with Large Experience data")
    plt.show()

# plot the rmse of critic loss for experiment 2 (small experience training data)
def plot_avg_rmse_2():
    plt.plot(np.arange(1, len(avg_rmse_q_loss_1) + 1), avg_rmse_q_loss_1, label='TL TD3')
    plt.plot(np.arange(1, len(avg_rmse_q_loss_2) + 1), avg_rmse_q_loss_2, label='Scratch TD3')
    plt.ylabel('Average RMSE Crtic Loss')
    plt.xlabel('Episode #')
    plt.legend(loc="lower right")
    plt.ylim(0, 3)
    plt.grid()
    plt.title("Average RMSE of critic loss with Small Experience Data")
    plt.savefig("Average RMSE of critic loss with Small Experience data")
    plt.show()



# Main function to call the plot function for different diagram
# Please uncomment the function name to plot relevant result
if __name__ == '__main__':
    plot_avg_reward()
    # plot_avg_rmse_1()
    # plot_avg_rmse_2()
    # best_score()
