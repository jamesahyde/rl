# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed April 29, 2020
adapted by jameshyde from
https://reinforcementlearning4.fun/2019/06/16/gym-tutorial-frozen-lake/

"""
# this is the Open AI Gym package that has built in environments, etc.
import gym

import argparse


def parse_args():
    # required line
    parser = argparse.ArgumentParser()

    # use add_argument for as many command line arguments you want to accept,
    # include data type and help text (default if desired)
    parser.add_argument('--run_type',
                        type=str,
                        default="n_actions",
                        help="how should this program run? 'n_actions' -> take specified number of random actions "
                             "'choose_each' -> manually select each action"
                        )
    parser.add_argument('--number',
                        type=int,
                        default="10",
                        help="how many actions should the agent take?"
                        )
    # required line
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

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

    action_dict = {"left": 0,
                   "down": 1,
                   "right": 2,
                   "up": 3
                   }

    if args.run_type == "n_actions":

        # number of actions to take
        MAX_ITERATIONS = args.num_actions

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
                print("TERMINAL STATE")
                print("Your Reward is {}".format(reward))
                if reward < 1:
                    print("...you fell in a hole")
                if reward == 1.0:
                    print("You got the frisbee!!!")
                break

    elif args.run_type == "choose_each":
        done = False
        while not done:
            action_string = input("choose a direction: ")
            action = action_dict[action_string]

            # .step method takes the chosen action and outputs the new state, reward from that action,
            # whether the episode is over or not, and meta info that is not used by the agent (for debugging)
            new_state, reward, done, info = env.step(action)
            # generate text-image of new environment and location of agent
            env.render()
            # if one of the actions results in a terminal state, done will be true, in which case exit the loop before 10 actions
            # terminal states are falling in a hole ("H") or getting to the goal ("G")

            if done:
                print("TERMINAL STATE")
                print("Your Reward is {}".format(reward))
                if reward < 1:
                    print("...you fell in a hole")
                if reward == 1.0:
                    print("You got the frisbee!!!")
                break