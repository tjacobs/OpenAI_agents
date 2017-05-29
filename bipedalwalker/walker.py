# OpenAI Gym Bipedal Walker implementation.
# By Tom Jacobs
# 
# Two methods:
# 1. Random: Tries random parameters, and picks the first one that gets a 200 score.
# 2. Mutation (Hill Climbing): Starts with random parameters, and adds a % mutation on the best parameters found.
#                              Mutation amount decays each episode until 0.
#
# Runs on Python 3.
# You can submit it to the OpenAI Gym scoreboard by entering your OpenAI API key into api_key.py and enabling submit below.

# Method to use?
method = 2

# Tunable params for method 2
episodes_per_update = 1 # Try each mutation with 20 episodes
mutation_amount = 1.0    # 100% random mutations each episode to start with
mutation_decay = 0.002    # How much we reduce mutation_amount by, each episode

# Submit it?
submit = True
import api_key

# Imports
import gym
import numpy as np
import math
import matplotlib.pyplot as plt

def run_episode(env, parameters, render=False):
    observation = env.reset()

    # Run 200 steps and see what our total reward is
    total_reward = 0
    timeOffset = 0
    for t in range(10000):

        # Hack to end the env if no good, otherwise Monitor freaks out
        if total_reward < 10 and t > 200:
            parameters[0] = 1
            parameters[1] = 1
            parameters[2] = 0
            parameters[3] = 1
            parameters[4] = 1
            parameters[5] = 0
            parameters[6] = 0
            parameters[7] = 1
            parameters[8] = 0
            parameters[9] = 0
            parameters[10] = 1
            parameters[11] = 0
            parameters[12] = 0.1

        if total_reward < 10 and t > 9500:
            render = True

        # Render. Uncomment this line to see each episode.
        if render:
            env.render()

        # Pick action
        action = [  math.sin(timeOffset + math.pi*parameters[0])*parameters[1] + parameters[2]/2, 
                    math.sin(timeOffset + math.pi*parameters[3])*parameters[4] + parameters[5]/2,
                    math.sin(timeOffset + math.pi*parameters[6])*parameters[7] + parameters[8]/2,
                    math.sin(timeOffset + math.pi*parameters[9])*parameters[10] + parameters[11]/2 ]
        timeOffset += parameters[12]

        # Step
        observation, reward, done, info = env.step(action)
        total_reward += reward

#        print( env.spec.timestep_limit, total_reward, t )

        # Done?
        if done:
            #print("Episode finished after {} timesteps".format(t+1))
            break
    return total_reward

def train(env):
    global mutation_amount

    # Keep results
    results = []

    # For method 1 and 2. Run lots of episodes with random params, and find the best_parameters.
    best_parameters = None
    best_reward = -1000 # Do better than falling backwards

    # Additional for method 2, start off with random parameters
    best_parameters = np.random.rand(13) * 2 - 1

    # Run
    for t in range(600):

        # Pick random parameters and run
        if method == 1:
            new_parameters = np.random.rand(13) * 2 - 1
            reward = run_episode(env, new_parameters)

        # Method 2 is to use the best parameters, with 10% random mutation
        elif method == 2:
            new_parameters = best_parameters + (np.random.rand(13) * 2 - 1) * mutation_amount
            mutation_amount = max(0.02, mutation_amount - mutation_decay)
            reward = 0
            for e in range(episodes_per_update):
                 run = run_episode(env, new_parameters)
                 reward += run
            reward /= episodes_per_update

        # One more result
        results.append(reward)

        # Did this one do better?
        print('Mutation amount: %.2f, Best: %.2f, Reward: %.2f' % (mutation_amount, best_reward, reward))
        if reward > best_reward:
            best_reward = reward
            best_parameters = new_parameters
            print("Better parameters found!")

            # And did we win the world?
            if reward >= 200:
                print("Win! Episode {}".format(t+1))
                break 

    # Run 100 runs with the best found params
    if submit:
        print("Found best_parameters, running 100 more episodes with them.")
        for t in range(100):
            reward = run_episode(env, best_parameters)
            results.append(reward)

    # Done
    return results, best_parameters


# Start cartpole
env = gym.make('BipedalWalker-v2')
env.spec.timestep_limit = 10
print( env.spec.timestep_limit )
if submit:
    env = gym.wrappers.Monitor(env, 'run', force=True)
env.spec.timestep_limit = 10
print( env.spec.timestep_limit )

# Try 10 times for a quick learn
for t in range(1):
    results, best_parameters = train(env)
    if submit:

        # Submit to OpenAI Gym if learned quickly enough
        reward = run_episode(env, best_parameters, True)
        print("Number of episodes run: {}".format(len(results)))
        if len(results) < 1000:
            print("Uploading to gym...")
            gym.scoreboard.api_key = api_key.api_key
            env.close()
            gym.upload('run')
            break

    else:
        for t in range(5):
            reward = run_episode(env, best_parameters, True)

        # Graph
        #plt.plot(results)
        #plt.xlabel('Episode')
        #plt.ylabel('Rewards')
        #plt.title('Rewards over time')
        #plt.show()


