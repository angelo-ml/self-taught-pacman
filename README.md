# Ms. Pac-Man Self Taught Algorithm
This algorithm implements reinforcement learning DQN technique to learn how to play Atari's Ms. Pac-Man, without any input data.

Deep Q-Network (DQN) combines a neural network, with the idea behind Q-learning, where an agent passes through different states multiple times, using and updating a Q-value function, which provides the rewards of different actions for each state. Each time the agent takes an action from a specific state and it receives its reward (or punishment), it updates the Q-value
function, using the bellman equation.

More details regarding the neural network used, the other optimization techniques implemented and the final results, are in the pdf file "project-report".

## Repository Files:
* train_pacman: trains the DQN model to play Ms. Pac-Man, collecting data from 10000 episodes.
* keras-learning-model.h5: the output trained, function approximation (a neural-network regressor model, using keras module).
* test_model: tests the DQN model for 100 episodes and stores the data of the episodes' scores in a csv. It also renders and records the last episode.
* last-episode.mp4: recorded video of the last testing episode


![Ms. Pac-Man Last Testing Episode](https://anjelo.ml/github-images/self-taught-pacman/pacman-last-episode.png)

