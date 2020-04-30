# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thurs April 30, 2020
adapted by jameshyde from
https://medium.com/swlh/introduction-to-reinforcement-learning-coding-q-learning-part-3-9778366a41c0

"""

import numpy as np
import time
# this is the Open AI Gym package that has built in environments, etc.
import gym



def choose_action(state):
    """"
    given a state, choose the next action to take
    """
    #action = 0
    # randomly draw from uniform distribution to determine if action should be random (probability epsilon) or not
    if np.random.uniform(0, 1) < epsilon:
        # if random action, sample from possible actions
        action = env.action_space.sample()
    else:
        # if not random, choose action based on Q-table such that state-action value is best possible, from this state
        action = np.argmax(Q[state, :])
    return action


def learn(state, state2, reward, action):
    """
    function to update Q table based on a given action
    :param state: current state
    :param state2: new state
    :param reward: reward from new state
    :param action: action being taken
    :return:
    """
    # predicted value of this action from this state
    predict = Q[state, action]
    # reward from this action, plus discounted reward from all future actions taken from new state
    # (as represented by the value of the highest value state-action pair for new state)
    target = reward + gamma * np.max(Q[state2, :])
    # update the current state-action value by adding current value to learning rate times difference between
    # true reward and predicted reward
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)

if __name__ == "__main__":


    #instantiate frozen lake environment
    env = gym.make('FrozenLake-v0')
    # # instantiate with deterministic state transitions
    # env = gym.make('FrozenLake-v0', is_slippery=False)
    # fraction of actions that should be random in 'epsilon-greedy' approach
    epsilon = 0.9
    total_episodes = 100000
    # maximum actions taken in a given episode
    max_steps = 100

    # learning rate
    lr_rate = 0.81
    # discount factor
    gamma = 0.96

    # create table of zeros with rows corresponding to possible states, and columns corresponding to possible actions
    # this will be updated to give value to state-action pairs as agent learns
    Q = np.zeros([env.observation_space.n,env.action_space.n])
    print("Beginning episodes with epsilon equal to {}".format(epsilon))
    all_eps = []
    # Start
    for episode in range(total_episodes):
        state = env.reset()
        t = 0
        ep_rew = 0
        if episode % 1000 == 0:
            print("Running episode #{}".format(episode))
        while t < max_steps:
            # # graphically render current state in the grid
            # env.render()
            # use custom function to choose an action based on the state and global epsilon value and Q table
            action = choose_action(state)
            # use openai gym function to take action and return new state (non-deterministic), reward, whether terminal
            # or not, and some metadata (like probability of action resulting in expected new state)
            state2, reward, done, info = env.step(action)
            # custom function to update Q table based on original state, new state, reward, and action taken
            learn(state, state2, reward, action)
            # set current state to be new state
            state = state2
            # add one to count of actions taken so far in this episode
            t += 1
            # add reward from this action to total reward accrued for this episode
            ep_rew += reward
            # end loop before max number of actions if this action resulted in a terminal state (i.e. done == True)
            if done:
                break
        # append total reward sum for this episode to list of episode-reward values
        all_eps.append(ep_rew)

    # print Q-table
    print(Q)
    # print average reward based on list of episode rewards
    print("Average reward over all episodes: {}".format(sum(all_eps) / len(all_eps)))

    ### re-run loop using different epsilon to allow policy to guide actions more often, but keep same Q table
    epsilon = 0.2
    total_episodes = 10000
    # maximum actions taken in a given episode
    max_steps = 100

    # learning rate
    lr_rate = 0.81
    # discount factor
    gamma = 0.96

    print("Beginning episodes with epsilon equal to {}".format(epsilon))
    all_eps = []
    # Start
    for episode in range(total_episodes):
        state = env.reset()
        t = 0
        ep_rew = 0
        if episode % 1000 == 0:
            print("Running episode #{}".format(episode))
        while t < max_steps:
            # # graphically render current state in the grid
            # env.render()
            # use custom function to choose an action based on the state and global epsilon value and Q table
            action = choose_action(state)
            # use openai gym function to take action and return new state (non-deterministic), reward, whether terminal
            # or not, and some metadata (like probability of action resulting in expected new state)
            state2, reward, done, info = env.step(action)
            # custom function to update Q table based on original state, new state, reward, and action taken
            learn(state, state2, reward, action)
            # set current state to be new state
            state = state2
            # add one to count of actions taken so far in this episode
            t += 1
            # add reward from this action to total reward accrued for this episode
            ep_rew += reward
            # end loop before max number of actions if this action resulted in a terminal state (i.e. done == True)
            if done:
                break
        # append total reward sum for this episode to list of episode-reward values
        all_eps.append(ep_rew)

    # print Q-table
    print(Q)
    # print average reward based on list of episode rewards
    print("Average reward over all episodes: {}".format(sum(all_eps) / len(all_eps)))