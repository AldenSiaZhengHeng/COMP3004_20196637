# This file will explain each of the file brieftly and shows how to run the project

# Github Repository: https://github.com/AldenSiaZhengHeng/COMP3004_20196637.git

# Thet project is run in below python libraries in MAC M1 Environment:
1. Python 3.10.5
2. torch 2.0.0
3. gym 0.26.2
4. matplotlib 3.7.1
5. numpy 1.24.3
6. tensorboard 2.12.3
7. Box2D 2.3.5

**To install the box2d gym environment: pip install 'gym[box2d]'**

-----------------------------------------
Preparation of environment to run the project:
1. The project is run on python virtual environment by entering:
    - **You can create your own virtual environment, this project is run on virtual environment that called td3**
    - 'python -m venv td3' #Create new virtual environment 
    - 'source td3/bin/activate' #This is to activate the virtual environment **(this is for MAC environment)**
    - extra for window env: '. td3/Scripts/activate'

2. Install the above libraries mentioned.

**If you are using window, if there is issue to install box2d library, normally install Microsoft C++ toolbox could fix it**

-----------------------------------------
# How to run the project
- **There is some adjustment might need to make like uncomment and comment certain function in different folder before run the file**
- **These adjustment had already state in each file as comment clearly, please reference to it**

1. Train the model in BipedalWalker Normal version: just run 'python train_standard.py'

2. To train the model for transfer learning in BipedalWalker Hardcore version: just run 'python train_hardcore_tl.py'

3. To train the model from scratch in BipedalWalker Hardcore version: just run 'python train_hardcore_scratch.py'

4. To test the model trained: just run 'python test.py'

5. To evaluate the training data of the model trained: just run 'pyhton utils.py'

6. To evaluate the result on tensorboard: 'tensorboard --logdir=runs' 
    - you can access the localhost website provided to evaluate the training result

-----------------------------------------
# Briefly explanation of each file and folder
- **The models, runs folder and the result that current contain in the main folder is trained on small experience data**

1. train_standard.py
    - this file is used to train the agent on normal version environment before transfer learning

2. train_hardcore_tl.py
    - this file will load the perfect pre-trained model and retrain again in hardcore version environment.

3. train_hardscore_scratch.py
    - this file will train the model from scratch on hardcore version environment.

4. utils.py
    - this file contain different plot function to plot the training result.

5. test.py
    - this file will run testing on the model train

6. TD3.py
    - this file contain the actor and critic network structure and TD3 algorithms function

7. ReplayBuffer.py
    - this file is used for storing the training data such as action, state, reward to train with TD3 algorithms.

8. models folder
    - all trained model that saved per intervals or achieve best result condition will be stored here
    - the optimzed model contain '_final_BipedalWalker-v3' word which achieve the best result condition
    - the rest that contain integer number such as 500, 1000, ... are the model that saved on each interval set.
    - it also contain the figure of train, test result

9. runs folder
    - this file contains the information stored to show on tensorboard

10. exp folder
    - this file contain the models, figure, result, data of 2 experiment which are train with large or small experience data
    - exp1 folder
        - this folder stored the data, models and others that trained with large experience data in hardcore environment
    - exp2 folder
        - this folder stored the data, models and others that trained with small experience data in hardcore environment
