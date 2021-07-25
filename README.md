# Ms. Pac-Man Self Taught Algorithm
This algorithm implements reinforcement learning DQN technique to learn how to play Atari's Ms. Pac-Man, without any input data.

Deep Q-Network (DQN) combines a neural network, with the idea behind Q-learning, where an agent passes through different states multiple times, using and updating a Q-value function, which provides the rewards of different actions for each state. Each time the agent takes an action from a specific state and it receives its reward (or punishment), it updates the Q-value function, using the bellman equation. 

More details regarding the optimization techniques implemented and the final results, are in the pdf file "project-report".

### Two different approaches are used

* In the first one, we used the game's image frames to train a Convolutional Neural Network.
* In the second one, we used game's RAM state to train a neural network consisted only by fully connected layers.

## Training the model using game's image frames

In this case the agent is trained based on a CNN that uses as data game's image frames (RGB 210x160px images). The input to the neural network consists of an 210x160x3 image. The first hidden layer convolves 32 filters of 8x8 with stride 4 with the input image and applies a rectifier nonlinearity. The second hidden layer convolves 64 filters of 4x4 with stride 2, again followed by a rectifier nonlinearity. This is followed by a third convolutional layer that convolves 64 filters of 3x3 with stride 1 followed by a rectifier. The final hidden layer is fully-connected and consists of 512 rectifier units. The output layer is a fully-connected linear layer with a single output for each valid action.

## Training the model using game's RAM state

Here, the RAM state is represented by an array of 128 integers between 0 and 255. The Neural network is consisted by three hidden fully-connected linear layers with 512, 512 and 128 rectifier units. Similarly with the CNN, the output layer is a fully-connected linear layer with a single output for each valid action.

## Repository Files

* train_pacman: trains the DQN model to play Ms. Pac-Man, collecting data from 10000 episodes.
* keras-learning-model.h5: the output trained, function approximation (a neural-network regressor model, using keras module).
* test_model: tests the DQN model for 100 episodes and stores the data of the episodes' scores in a csv. It also renders and records the last episode.
* last-episode.mp4: recorded video of the last testing episode


![Ms. Pac-Man Last Testing Episode](https://anjelo.ml/github-images/self-taught-pacman/pacman-last-episode.png)

