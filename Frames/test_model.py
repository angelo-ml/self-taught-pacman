"""
Training Ms. Pac-Man using openAI Gym
@author: Angelos Malandrakis
"""


# load libraries
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas as pd
import gym
import time


# %% load the game environment
env = gym.make('MsPacman-v4')
model = load_model('keras-learning-model.h5')

# %% Initialize variables
states = env.observation_space.shape[0]
observation_space = env.observation_space.shape
batch_observation_space = (1,) + observation_space
actions = env.action_space.n
render = False
sleeping_time = 0.0


# %% set parameters
episodes = 100



# %% test game
scores_test = []
for episode in range(1, (episodes+1)):
	
	# If it's the last episode, it renders and records the game
	if episode == episodes:
		env = gym.wrappers.Monitor(env,'Videos/openaiGym/',force=True)
		render = True
		sleeping_time = 0.03
		
	# at the beginning of each episode, reset the environment	
	state = env.reset()
	
	# reset the values bellow
	done = False
	time_step = 0
	episode_reward = 0

	while not done:
		if render:
			env.render()
		
		# predict the best action
		action = np.argmax(model.predict(np.reshape(state,batch_observation_space))[0])
		time.sleep(sleeping_time)
		
		# take the action 
		new_state, reward, done, info = env.step(action)
		# save the new state as state
		state = new_state
		
		# add to the episode total score
		episode_reward += reward
	
	# save episode's score to the list with episodes' scores 
	scores_test.append(episode_reward)
print('Average score over the games: {}'.format(np.mean(scores_test)))

# save game scores
score_data = pd.DataFrame(scores_test)
score_data.to_csv('test_scores.csv', sep=";", index=False, header=False)