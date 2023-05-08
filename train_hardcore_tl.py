# his file is used to retrain the TD3 agent by transfer learning from pre-trained agent in BipedalWalker-v3 Hardcore version

# Import library 
import numpy as np
import torch
import gym
from TD3 import TD3
import matplotlib.pyplot as plt
import ReplayBuffer
import os
import csv
import time
import math
from torch.utils.tensorboard import SummaryWriter

# This is used to allocated and move data between CPU and GPU for memory allocation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main function to train the model for scratch
def main(seed):
    env_with_Dead = True  # Whether the Env has dead state. True for Env like BipedalWalkerHardcore-v3, CartPole-v0. False for Env like Pendulum-v0

    # Set the type of the game to play
    env_id = 'BipedalWalker-v3'

    # Uncomment this and comment the other make function if you want to evaluate the game
    # A frame will pop out and show the game play
    # env = gym.make(env_id, render_mode="human", hardcore=True) # This function will create the game play environment with display frame

    env = gym.make(env_id, hardcore=True) # This function will create the game play environment
    
    # Obtain the state, action and maximum action space
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Explosion noise (to prevent the agent stuck in local minima)
    expl_noise = 0.25

    # Show the details of action, state,...
    print('  state_dim:', state_dim, '  action_dim:', action_dim, '  max_a:', max_action, '  min_a:', env.action_space.low[0])

    # Variable for load model, set random seed, maximum episode for training and save interval for saving the training model per amoount of episode
    Loadmodel = True
    random_seed = seed

    # Variable for load model, set random seed, maximum episode for training and save interval for saving the training model per amoount of episode
    Max_episode = 3000
    save_interval = 500 #interval to save model

    # Create the file path if not exists to store record
    models_path = 'models'
    runs_path = 'runs/exp_tl_hardcore'


    # Create the file path if not exists to store record
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    if not os.path.exists(runs_path):
        os.makedirs(runs_path)

    # Set the random seed to project environment
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # Set the writer to plot data into tensorboard
    writer = SummaryWriter(log_dir='runs/exp_tl_hardcore')


    # Set the hyperparameter settings for TD3
    kwargs = {
        "env_with_Dead":env_with_Dead,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": 0.99,
        "net_width": 200,
        "a_lr": 5e-5,
        "c_lr": 5e-5,
        # lower learning rate for fine tuning
        # "a_lr": 4e-3,
        # "c_lr": 4e-3,
        "Q_batchsize":256,
    }
    model = TD3(**kwargs)

    # Function to load the model
    if Loadmodel: model.load('_final_' + env_id)

    # Set the constructor with the replay buffer size, dimensional of state and action
    replay_buffer = ReplayBuffer.ReplayBuffer(state_dim, action_dim, max_size=int(1e6))

    # Initialize and set the variable to record the data during traninig
    all_ep_r, avg_score, scores = [], [], []
    best_score = -np.inf
    start_time = time.time()

    # Set maximum timestep to run experiment in each episode
    max_timestep = 3000

    # Main loop to run the experiment
    for episode in range(Max_episode):

        # Reset the environment at the beginning and everytime per episode end
        s, done = env.reset(), False
        if len(s) == 2:
            s = np.array(s[0]) 
        ep_r = 0
        steps = 0
        expl_noise *= 0.999

        # to record episode per minutes
        EPM = 0

        # To reach total time taken
        TTM = 0
        t0 = time.time()

        rmse_q_loss = []

        # Train when it is not done or haven't reach maximum timesteps set.
        while not done and steps < max_timestep:
            steps+=1

            # Get the action from TD3 by adding noise value
            a = ( model.select_action(s) + np.random.normal(0, max_action * expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

            # Received state, reward, done and other information after apply the step to the robot agent
            s_prime, r, done, truncated, info = env.step(a)

            # Tricks for BipedalWalker
            # Reward Shapping to prevent huge gap on Q-value for stable training 
            if r <= -100:
                r = -1
                replay_buffer.add(s, a, r, s_prime, True)
            else:
                replay_buffer.add(s, a, r, s_prime, False)

            # Only train when gathered enough data in replay buffer
            # Modify here to adjust whether to train the agent with large of small experience data
            # Just uncomment this and commet the below one to change from small to large experience data

            # if replay_buffer.size > 8000: #large experience data
            #     model.train(replay_buffer)

            if replay_buffer.size > 2000: # small experience data
                model.train(replay_buffer)

            s = s_prime
            ep_r += r
        # Store the score and calculate the average score obtained
        scores.append(ep_r)
        avg_score = np.mean(scores[-100:])

        # Calculate the rmse of critic loss
        rmse_q_loss = model.return_rmse_q_loss()
        avg_rmse_q_loss = np.mean(rmse_q_loss)
        model.reset_rmse_q_loss()

        print(f'| Game: {episode:6.0f} | Score: {ep_r:10.2f} | Best score: {best_score:10.2f} | '
              f'Avg score {avg_score:10.2f} | Avg RMSE Q loss: {avg_rmse_q_loss:10.3f}')

        if avg_score > best_score:
            best_score = avg_score
        #     model.save("_final_" + env_id)

        # Save the model per interval set
        if (episode+1)%save_interval==0:
            model.save(str(episode + 1) + "-tl-hardcore")
        
        # If the score obtain by agent can consistency larger than 300 within range of 50, then save the agent as best agent and stop training
        if(np.mean(scores[-20:]) >= 300):
            model.save("_final_" + env_id + "-tl-hardcore")
            break


        # Record the training log on tensorboard
        if episode == 0: all_ep_r.append(ep_r)
        else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
        EPM = int(time.time() - t0)
        TTM = int((time.time() - start_time)/60)
        writer.add_scalar('s_ep_r', all_ep_r[-1], global_step=episode)
        writer.add_scalar('ep_r', ep_r, global_step=episode)
        writer.add_scalar('exploare', expl_noise, global_step=episode)
        writer.add_scalar('Episode Per Minutes',EPM, global_step=episode)
        writer.add_scalar('Time Taken (Minutes)', TTM, global_step=episode)
        writer.add_scalar('Average Reward', avg_score, global_step=episode)
        writer.add_scalar('Time step For each episode', steps, global_step=episode)
        writer.add_scalar('Average RMSE Q loss', avg_rmse_q_loss, global_step=episode)
        print('seed:',random_seed,'episode:', episode,'score:', ep_r, 'step:',steps , 'max:', max(all_ep_r))

        # save progress into csv file
        fields = [episode, ep_r, avg_score, EPM, TTM, avg_rmse_q_loss, steps]
        with open(env_id + "-tl-hardcore-data.csv", "a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(fields)


    # close the environment once finish training
    env.close()

    # Display and plot Scores
    avg_x = [np.mean(scores[np.max([0, i - 100]):i]) for i in range(len(scores))]
    #Display Scores
    plt.figure(dpi=200)
    plt.title(env_id + '-tl-Hardcore')
    plt.plot(np.arange(1, len(scores) + 1), scores, label='score')
    plt.plot(range(len(avg_x)), avg_x, label="average score")
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.ylim(-300, 300)
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("models/" + 'train_tl_BipedalWalker-v3-hardcore.png')
    plt.show() 


# Main function to run and the seed is set as 1
if __name__ == '__main__':
    main(seed=1)
    





