# OpenAI Gym Bipedal Walker implementation.
# By Tom Jacobs
# 
# You can submit it to the OpenAI Gym scoreboard by entering your OpenAI API key into api_key.py and enabling submit below.

# Submit it?
submit = False
import api_key

# Imports
import gym
import numpy as np
import math

NUM_PARAMETERS = 5

def run_episode(env, parameters, render=False):
    state = env.reset()

    # Tunable constants
    SPEED = 1.0

    # States
    LEFT_FOOT_STANCE, RIGHT_FOOT_DOWN, RIGHT_FOOT_STANCE, LEFT_FOOT_DOWN = 1, 2, 3, 4
    walk_state = LEFT_FOOT_STANCE
    left_leg   = 0
    right_leg = 1

    # Counts
    steps = 0
    total_reward = 0
    time = 0

    # Loop
    for t in range(1000):

        # Render
        if render:
            env.render()

        # Which indicies are the angles for the legs
        left_hip_angle_index      = 4
        right_leg_hip_angle_index = 9

        # Targets
        hip_target  = [None, None]   # Hips go  -0.8 to 1.1
        knee_target = [None, None]   # Knees go -0.6 to 0.9

        # Timer
        time += 1
        time_mod = time % 500

        # State to target mapping
        if walk_state == LEFT_FOOT_STANCE:
            # Move left leg forward
            hip_target[left_leg]  = 1.1
            knee_target[left_leg] = -0.6

            hip_target[right_leg]  = -0.2
            knee_target[right_leg] = -0.3

            # Jump!
            if time_mod > 100 and time_mod < 200:
                knee_target[right_leg] = 0.15
                knee_target[left_leg] = -0.15

                # If both legs off the ground, go for the leg flip
                if not state[8] and not state[13]:
                    print( "Flip!" )
                    walk_state = RIGHT_FOOT_STANCE
                    time = 200

        elif walk_state == RIGHT_FOOT_DOWN:
            if state[8]:
                print( "Right foot is down." )
                walk_state = RIGHT_FOOT_STANCE

        elif walk_state == RIGHT_FOOT_STANCE:
            # Move right leg forward
            hip_target[right_leg]  = 1.1
            knee_target[right_leg] = -0.6

            hip_target[left_leg]  = -0.2
            knee_target[left_leg] = -0.3

            # Jump!
            if time_mod > 100 and time_mod < 200:
                knee_target[left_leg] = 0.15
                knee_target[right_leg] = -0.15

                # If both legs off the ground, go for the leg flip
                if not state[8] and not state[13]:
                    print( "Flip!" )
                    walk_state = LEFT_FOOT_STANCE
                    time = 200

        elif walk_state == LEFT_FOOT_DOWN:
            if state[13]:
                print( "Left foot is down." )
                walk_state = LEFT_FOOT_STANCE

        # How should we move?
        hip_movement  = [0.0, 0.0]
        knee_movement = [0.0, 0.0]

        # If we have targets, use PD controller to move them there. Kp * anglediff - Kd * velocity
        Kp = 25.0
        Kd = 0.1
        hip_in_world_frame = (state[4] - state[0], state[9] - state[0])
        hip_velocity_in_world_frame = (state[5] - state[1], state[10] - state[1])
        if hip_target[0]:  hip_movement[0]  = Kp * (hip_target[0]  - hip_in_world_frame[0])  + Kd * hip_velocity_in_world_frame[0]
        if knee_target[0]: knee_movement[0] = Kp * (knee_target[0] - state[6])               + Kd * state[7]
        if hip_target[1]:  hip_movement[1]  = Kp * (hip_target[1]  - hip_in_world_frame[1])  + Kd * hip_velocity_in_world_frame[1]
        if knee_target[1]: knee_movement[1] = Kp * (knee_target[1] - state[11])              + Kd * state[12]

        # Balance. PD controller to adjust hip in world frame to keep balance up straight. Kp * body_angle + Kd * body_angle_velocity.
        kBalanceP = 50.0
        kBalanceD = 5
        hip_movement[0] += kBalanceP * state[0] + kBalanceD * state[1]
        hip_movement[1] += kBalanceP * state[0] + kBalanceD * state[1]

        # Dampen bounce. PD controller to adjust knee according to vertical speed, to dampen bouncy oscillations. Kd*body_vertical_velocity.
        knee_movement[0] -= 25.0 * state[3]
        knee_movement[1] -= 25.0 * state[3]

        # Set action
        action = np.array([hip_movement[0], knee_movement[0], hip_movement[1], knee_movement[1]])
        action = np.clip(action, -1.0, 1.0)
        #print( "Torques: Hip %0.2f Knee %0.2f      Hip %0.2f Knee %0.2f" % (action[0], action[1], action[2], action[3] ) )

        # Step
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        # Display
        if steps % 200 == 0 or done:
            print("Step {} total_reward {:+0.2f}".format(steps, total_reward))
            print("Hull " + str(["{:+0.2f}".format(x) for x in state[0:4] ]))
            print("Leg 0 " + str(["{:+0.2f}".format(x) for x in state[4:9] ]))
            print("Leg 1 " + str(["{:+0.2f}".format(x) for x in state[9:14]]))

        # Done?
        if done:
            break
    return total_reward

def train(env, best_parameters):
    global mutation_amount

    # Keep results
    results = []
    best_reward = -100 # Do better than falling down

    # Try some episodes
    for t in range(1):
        # Try new paremeters
        new_parameters = best_parameters + (np.random.rand(NUM_PARAMETERS) * 2 - 1) * mutation_amount
        mutation_amount = max(mutation_min, mutation_amount - mutation_decay)
        reward = 0
        for e in range(episodes_per_update):
             run = run_episode(env, new_parameters, not submit)
             reward += run
        reward /= episodes_per_update

        # One more result
        results.append(reward)

        # Display
        mutation_display = int(mutation_amount * 1000)
        if mutation_display % 10 == 9 or mutation_display % 10 == 0:
            print('Mutation amount: %.2f.' % (mutation_amount))

        # Did this one do better?            
        if reward > best_reward:
            best_reward = reward
            best_parameters = new_parameters
            print("Better parameters found. Best reward so far: %d" % best_reward)

            # And did we win the world?
            if reward >= 300:
                print("Win! Episode {}".format(t+1))
                mutation_amount = 0.01
                break 

    # Done
    return results, best_parameters, best_reward

# Start
env = gym.make('BipedalWalker-v2')
if submit:
    env = gym.wrappers.Monitor(env, 'run', force=True)

# Record the best of the best
best_best_parameters = np.random.rand(NUM_PARAMETERS) * 2 - 1
best_best_reward = -100
for t in range(1):
    print('\nTraining run {}/2'.format(t+1))

    episodes_per_update = 1  # Try each mutation with a few episodes
    mutation_amount = 0.1    # Random mutations each episode to start with
    mutation_decay = 0.01    # How much we reduce mutation_amount by, each episode
    mutation_min = 0.01      # Keep mutating at the end by this much

    results, parameters, best_reward = train(env, best_best_parameters)
    if best_reward > best_best_reward:
        best_best_parameters = parameters
        best_best_reward = best_reward

# Run odd looking Forest, run
#print('We ended up running like this. Parameters: Speed: %.2fx, %.2f, %.2f, %.2f, %.2f. Best best reward: %d' % (best_best_parameters[0], best_best_parameters[1], best_best_parameters[2], best_best_parameters[3], best_best_parameters[4], best_best_reward) )
#reward = run_episode(env, best_best_parameters, True)

# Submit
if submit and best_best_reward > 100:
        # Run 100 runs with the best found params
        print("Found best_best_parameters, running 100 more episodes with them.")
        for t in range(100):
            reward = run_episode(env, best_best_parameters)
            results.append(reward)

        # Submit to OpenAI Gym if learned quickly enough
        print("Uploading to gym...")
        gym.scoreboard.api_key = api_key.api_key
        env.close()
        gym.upload('run')
