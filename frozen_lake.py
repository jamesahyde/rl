# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed April 29, 2020
adapted by jameshyde from
https://reinforcementlearning4.fun/2019/06/16/gym-tutorial-frozen-lake/

"""

# this is the Open AI Gym package that has built in environments, etc.
import gym

# number of actions to take
MAX_ITERATIONS = 10

# instantiate frozen lake environment
env = gym.make("FrozenLake-v0")

# display the action space as defined in the frozen lake environment
print("Action space: ", env.action_space)
# display the observation space as defined in the frozen lake environment
print("Observation space: ", env.observation_space)

# reset the environment
env.reset()
# generate text-image of environment and location of agent
env.render()
# loop through number of actions
for i in range(MAX_ITERATIONS):
    # randomly choose one of the available actions (actions are non-deterministic so chosen action might not be implemented)
    random_action = env.action_space.sample()
    # .step method takes the chosen action and outputs the new state, reward from that action,
    # whether the episode is over or not, and meta info that is not used by the agent (for debugging)
    new_state, reward, done, info = env.step(random_action)
    # generate text-image of new environment and location of agent
    env.render()
    # if one of the actions results in a terminal state, done will be true, in which case exit the loop before 10 actions
    # terminal states are falling in a hole ("H") or getting to the goal ("G")
    if done:
        break


