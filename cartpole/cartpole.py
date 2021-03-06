# OpenAI Gym Cartpole implementations.
# By Tom Jacobs
# 
# Two methods:
# 1. Random: Tries random parameters, and picks the first one that gets a 200 score.
# 2. Mutation (Hill Climbing): Starts with random parameters, and adds a % mutation on the best parameters found.
#                              Mutation amount decays each episode until 0.
#
# Runs on Python 3.
# Originally based on https://github.com/kvfrans/openai-cartpole
# You can submit it to the OpenAI Gym scoreboard by entering your OpenAI API key into api_key.py and enabling submit below.

# Method to use?
method = 1

# Tunable params for method 2
episodes_per_update = 20 # Try each mutation with 20 episodes
mutation_amount = 1.0    # 100% random mutations each episode to start with
mutation_decay = 0.01    # How much we reduce mutation_amount by, each episode

# Submit it?
submit = True
import api_key

# Imports
import gym
import numpy as np
import matplotlib.pyplot as plt

def run_episode(env, parameters):
    observation = env.reset()

    # Run 200 steps and see what our total reward is
    total_reward = 0
    for t in range(200):

        # Render. Uncomment this line to see each episode.
#        env.render()

        # Pick action
        action = 0 if np.matmul(parameters, observation) < 0 else 1

        # Step
        observation, reward, done, info = env.step(action)
        total_reward += reward

        # Done?
        if done:
            #print("Episode finished after {} timesteps".format(t+1))
            break
    return total_reward

def train(submit):
    global mutation_amount

    # Start cartpole
    env = gym.make('CartPole-v0')
    if submit:
        env = gym.wrappers.Monitor(env, 'run-cartpole', force=True)

    # Keep results
    results = []

    # For method 1 and 2. Run lots of episodes with random params, and find the best_parameters.
    best_parameters = None
    best_reward = 0

    # Additional for method 2, start off with random parameters
    best_parameters = np.random.rand(4) * 2 - 1

    # Run
    for t in range(100):

        # Pick random parameters and run
        if method == 1:
            new_parameters = np.random.rand(4) * 2 - 1
            reward = run_episode(env, new_parameters)

        # Method 2 is to use the best parameters, with 10% random mutation
        elif method == 2:
            new_parameters = best_parameters + (np.random.rand(4) * 2 - 1) * mutation_amount
            mutation_amount = max(0, mutation_amount - mutation_decay)
            reward = 0
            for e in range(episodes_per_update):
                 run = run_episode(env, new_parameters)
                 reward += run
            reward /= episodes_per_update

        # One more result
        results.append(reward)

        # Did this one do better?
        if reward > best_reward:
            best_reward = reward
            best_parameters = new_parameters
            print("Better parameters found.")

            # And did we win the world?
            if reward == 200:
                print("Win! Episode {}".format(t+1))
                break # Can't do better than 200 reward, so quit trying

    # Run 100 runs with the best found params
    print("Found best_parameters, running 100 more episodes with them.")
    for t in range(100):
        reward = run_episode(env, best_parameters)
        results.append(reward)
        #print( "Episode " + str(t) )

    # Done
    return results

# Try 10 times for a quick learn
for t in range(10):
    results = train(submit=submit)
    if submit:
        # Submit to OpenAI Gym if learned quickly enough
        print("Number of episodes run: {}".format(len(results)))
        if len(results) < 120:
            print("Uploading to gym...")
            gym.scoreboard.api_key = api_key.api_key
            gym.upload('run-cartpole')
            break

    else:
        # Graph
        plt.plot(results)
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.title('Rewards over time')
        plt.show()


