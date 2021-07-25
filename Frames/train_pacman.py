"""
Created on Mon May  4 16:52:58 2020
@author: Angelos Malandrakis
"""

# load libraries
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as k
from collections import deque
import gym
import pandas as pd
import time


## Initialize Approximation Functions
def approximationFunction(observation_space, actions):
	"""Deep Neural Network."""
	model = Sequential()
	# first convolutional layer
	model.add(Conv2D(32, kernel_size=8, strides=(4, 4), activation='relu', input_shape=observation_space))
	# second convolutional layer
	model.add(Conv2D(64, kernel_size=4, strides=(2, 2), activation='relu'))
	# third convolutional layer
	model.add(Conv2D(64, kernel_size=3, strides=(1,1), activation='relu'))
	# Fully connected layers
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dense(actions, activation='relu'))
	return model




# load game's environment
env = gym.make('MsPacman-v4')

# Set seed for reproducibility
seed_val = 456
np.random.seed(seed_val)
env.seed(seed_val)
random.seed(seed_val)

observation_space = env.observation_space.shape
batch_observation_space = (1,) + observation_space
actions = env.action_space.n

# Discount in Bellman Equation
gamma = 0.95
# Epsilon
epsilon = 1.0
# Minimum Epsilon
epsilon_min = 0.05
# Decay multiplier for epsilon
epsilon_decay = 0.9995
# Size of training memory
deque_len = 100000
# Average score needed over 100 epochs
target_score = 1000
# Number of games
episodes = 5
# Data points per episode used to train the agent
batch_size = 32
# Optimizer for training the agent
learning_rate = 0.0001
# Loss for training the agent
loss = 'mse'


training_data = deque(maxlen=deque_len)

print('----Training----')

starting_time = time.time()
starting_CPUtime = time.process_time()

k.clear_session()

# define empty list to store the score at the end of each episode
scores = []
avg_scores = []
# load the agent
model = approximationFunction(observation_space, actions)
# compile the agent with mean squared error loss
model.compile(loss=loss, optimizer=Adam(learning_rate=learning_rate))


for episode in range(1, (episodes+1)):
	state = env.reset()
	done = False
	time_step = 0
	episode_score = 0
	
	
	while not done:
		
		#select action
		if np.random.rand() <= epsilon:
			action = random.randrange(actions)
		else:
			action = np.argmax(model.predict(np.reshape(state,batch_observation_space))[0])
			
		# take the action
		new_state, reward, done, info = env.step(action)
		
		if done:
			action_reward = -150
		elif reward == 0: 
			action_reward = -2
		else:
			action_reward = reward
			
		# traing the CNN with the new data	
		training_data.append((state, new_state, action_reward, done, action))
		
		# set state to new state
		state = new_state
		# increment timestep
		time_step += 1
		episode_score += reward
	
	# Train Approximation Function
	idx = random.sample(range(len(training_data)), min(len(training_data), batch_size))
	train_batch = [training_data[j] for j in idx]
	for state, new_state, reward, done, action in train_batch:
		target = reward
		if not done:
			target = reward+gamma*np.amax(model.predict(np.reshape(new_state,batch_observation_space))[0])
		#print('target', target)
		target_f = model.predict(np.reshape(state,batch_observation_space))
		#print('target_f', target_f)
		target_f[0][action] = target
		#print('target_f_r', target_f)
				
		model.fit(np.reshape(state,batch_observation_space), target_f, epochs=1, verbose=0)
	if epsilon > epsilon_min:
		epsilon *= epsilon_decay
		epsilon = max(epsilon_min, epsilon)
		
	scores.append(episode_score)

	if episode % 100 == 0:
		print('episode {}, score {}, epsilon {:.4}'.format(episode,episode_score,epsilon))
		print('Avg Score over last 100 epochs', sum(scores[-100:])/100)
		avg_scores.append(sum(scores[-100:])/100)
		
		if sum(scores[-100:])/100 > target_score:
			print('------ Score Achieved After {} Episodes ------'.format(episode))

# Calculate running time
total_time = round(time.time() - starting_time,2)
cpu_time = round(time.process_time() - starting_CPUtime,2)
print('========= Total Time: {} || CPU Time: {} ========='.format(total_time, cpu_time))
			
plt.figure(figsize=(20, 5))
plt.title('Plot of Avg. Score v/s 100 Episodes')
plt.xlabel('Episodes x 100')
plt.ylabel('Avg. Scores')
plt.plot(avg_scores)
plt.grid(True)
plt.savefig('score-long.png')

plt.figure()
plt.title('Plot of Avg. Score v/s 100 Episodes')
plt.xlabel('Episodes x 100')
plt.ylabel('Avg. Scores')
plt.plot(avg_scores)
plt.grid(True)
plt.savefig('score.png')

scoresDataframe = pd.DataFrame(scores)
scoresDataframe.to_csv('scores.csv', sep=";", index=False, header=False)

avg_scoresDataframe = pd.DataFrame(avg_scores)
avg_scoresDataframe.to_csv('avg_scores.csv', sep=";", index=False, header=False)

model.save('keras-learning-model.h5')
