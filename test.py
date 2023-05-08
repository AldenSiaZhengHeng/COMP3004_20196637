# This file is used to evaluate the performance of the trained model

import numpy as np
import torch
import gym
from TD3 import TD3
import matplotlib.pyplot as plt
import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

# This is used to allocated and move data between CPU and GPU for memory allocation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main function to perform testing
def main(seed):
    env_with_Dead = True  # Whether the Env has dead state. True for Env like BipedalWalkerHardcore-v3, CartPole-v0. False for Env like Pendulum-v0

    # Set the type of the game to play
    env_id = 'BipedalWalker-v3'


    # Uncomment this and comment the other make function if you want to evaluate the game
    # A frame will pop out and show the game play
    # env = gym.make(env_id, render_mode="human", hardcore=True) # This is for hardcore version
    # env = gym.make(env_id, render_mode="human") # This is for normal version

    # This function will create the game play environment
    env = gym.make(env_id, hardcore=True) # Hardcore Version
    # env = gym.make(env_id) # Normal Version

    # Obtain the state, action and maximum action space
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    expl_noise = 0.25
    print('  state_dim:', state_dim, '  action_dim:', action_dim, '  max_a:', max_action, '  min_a:', env.action_space.low[0])

    # Variable for load model, set random seed, maximum episode for training and save interval for saving the training model per amoount of episode
    Loadmodel = True
    ModelIdex = 3000 #which model to load
    random_seed = seed

    Max_episode = 100

    # Set the random seed to project environment
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        # env.seed(random_seed)
        np.random.seed(random_seed)

    # Set the hyperparameter settings for TD3
    kwargs = {
        "env_with_Dead":env_with_Dead,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": 0.99,
        "net_width": 200,
        "a_lr": 1e-4,
        "c_lr": 1e-4,
        "Q_batchsize":256,
    }
    model = TD3(**kwargs)

    ## This section is used to select which model to load
    # Uncomment one of these if there is optimal model generated from training that achieve the best result condition

    # if Loadmodel: 
    #     model.load('_final_' + env_id) # Load model that trained in normal version
    # if Loadmodel: 
    #     model.load('_final_' + env_id+ "-tl-hardcore") # Load TL model
    # if Loadmodel: 
    #     model.load('_final_' + env_id +"-hardcore-scratch") # Load Sractch model

    ## This section is for the model that trained for maximum episode that does not achieve the best result condition and train until the end
    ## You can adjust the "ModelIdex" number at above to choose to load different model that saved in different interval
    if Loadmodel: 
        model.load(str(ModelIdex) + "-tl-hardcore") # Load TL model

    ## Uncomment this and comment other if you want to test the scratch model
    # if Loadmodel: 
    #     model.load(str(ModelIdex) + "-hardcore-scratch") # Load scratch model

    # Initialize and set the variable to record the data during traninig
    all_ep_r, avg_score, scores = [], [], []
    best_score = -np.inf
    max_timestep = 2000

    print("Start testing for " + str(Max_episode) + "episode......")

    # Main loop to run the experiment
    for episode in range(Max_episode):
        s, done = env.reset(), False
        if len(s) == 2:
            s = np.array(s[0]) 
        ep_r = 0
        steps = 0
        expl_noise *= 0.999

        while not done and steps < max_timestep:
            steps+=1

            a = model.select_action(s)
            s_prime, r, done, truncated, info = env.step(a)

            if r <= -100:
                r = -1

            s = s_prime
            ep_r += r
        scores.append(ep_r)
        avg_score = np.mean(scores[-100:])



        print(f'| Game: {episode:6.0f} | Score: {ep_r:10.2f} | Best score: {best_score:10.2f} | '
              f'Avg score {avg_score:10.2f} | Steps: {steps}')

        if avg_score > best_score:
            best_score = avg_score
        #     model.save(episode + 1)


    # close the environment once finish training
    env.close()

    # Display and plot Scores
    avg_x = [np.mean(scores[np.max([0, i - 100]):i]) for i in range(len(scores))]
    #Display Scores
    plt.figure()
    plt.plot(np.arange(1, len(scores) + 1), scores, label='scores')
    plt.plot(range(len(avg_x)), avg_x, label='average score')
    plt.ylabel('Score')
    plt.xlabel('Episode #')

    ## This is to plot three different graph for standard, TL and scratch model
    # plt.title(env_id + "-standard")
    plt.title(env_id + "-tl-hardcore")
    # plt.title(env_id + "-hardcore-scratch")
    plt.legend(loc="lower right")
    plt.grid()

    # Function to save the figure
    ## Uncommet to specific one to saved the figure
    plt.savefig("models/" + 'test_tl_BipedalWalker-v3-hardcore.png')
    # plt.savefig("models/" + 'test_BipedalWalker-v3-hardcore-scratch.png')
    # plt.savefig("models/" + 'test_BipedalWalker-v3-standard.png')
    plt.show()




if __name__ == '__main__':
    main(seed=1)
    





