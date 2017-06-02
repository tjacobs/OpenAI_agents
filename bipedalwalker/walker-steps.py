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
    observation = env.reset()

    SPEED = 0.4 + parameters[0]/5
    SUPPORT_KNEE_ANGLE = 0.1

    # Action to take
    a = np.array([0.0, 0.0, 0.0, 0.0])
    steps = 0
    total_reward = 0

    # States
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF, STEP_BACK, STEP_FORWARDS = 1, 2, 3, 4, 5
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    supporting_knee_angle = SUPPORT_KNEE_ANGLE

    for t in range(10000):

        # Step
        s, r, done, info = env.step(a)
        total_reward += r

        # Render
        if render:
            env.render()

        # Display
        if False: #steps % 20 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
            print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
            print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
        steps += 1

        # Foot touched the ground?
        contact0 = s[8]
        contact1 = s[13]

        # Which indicies are the angles for the legs
        moving_s_base     = 4 + 5*moving_leg
        supporting_s_base = 4 + 5*supporting_leg

        # Targets
        hip_target  = [None, None]   # -0.8 .. +1.1
        knee_target = [None, None]   # -0.6 .. +0.9

        # What actions should we take
        hip_todo  = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        # State to target mapping
        if state==STEP_BACK:
            if s[2] < -0.2:
                hip_target[moving_leg] = 1.0
                knee_target[moving_leg] = 1.0

            if s[2] > 0.1:
                print( "We're ok now.")
                state = STAY_ON_ONE_LEG

        if state==STEP_FORWARDS:
            if s[2] > 0.5:
                pass
            if s[2] < 0.01:
                print( "We're ok now.")
                state = STAY_ON_ONE_LEG

        if state==STAY_ON_ONE_LEG:
            # Move moving leg
            hip_target[moving_leg]  = 1.2
            knee_target[moving_leg] = -0.7
            supporting_knee_angle += 0.04

            # Bolt forward if falling forward
            if s[2] > SPEED: supporting_knee_angle += 0.10

            # Angle knee of supporting leg
            supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
            knee_target[supporting_leg] = supporting_knee_angle

            # If the supporting leg is way behind now, let's go!
            if s[supporting_s_base] < 0.20: 
                state = PUT_OTHER_DOWN

            # Oop, are we falling backwards?
            if s[2] < -0.2:
                print( "Falling backwards.")
                render = True
                state = STEP_BACK

            # Oop, are we falling forwards?
            if s[2] > 0.5:
                print( "Falling forwards. {}".format(s[2]))
                render = True
#                state = STEP_FORWARDS

        if state==PUT_OTHER_DOWN:
            # Put our foot down. Moving leg now becomes supporting leg.
            hip_target[moving_leg]      = 0.1
            knee_target[moving_leg]     = SUPPORT_KNEE_ANGLE
            knee_target[supporting_leg] = supporting_knee_angle

            # When it's time, push off
            if s[moving_s_base+4]:
                state = PUSH_OFF
                supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )

        if state==PUSH_OFF:
            # Just bounce
            knee_target[moving_leg] = supporting_knee_angle
            knee_target[supporting_leg] = 1.0

            # Did we bounce?
            if s[supporting_s_base+2] > 0.88 + parameters[1]/5 or s[2] > 1.2*SPEED*parameters[2]/5:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        # Calculate movements to move towards targets
        if hip_target[0]:  hip_todo[0]  = (parameters[3]/5 + 0.9) * (hip_target[0]  - s[4])  - 0.25*s[5]
        if hip_target[1]:  hip_todo[1]  = (parameters[3]/5 + 0.9) * (hip_target[1]  - s[9])  - 0.25*s[10]
        if knee_target[0]: knee_todo[0] = (parameters[4]/5 + 4.0) * (knee_target[0] - s[6])  - 0.25*s[7]
        if knee_target[1]: knee_todo[1] = (parameters[4]/5 + 4.0) * (knee_target[1] - s[11]) - 0.25*s[12]

        # Adjust movement to stand up straight
        hip_todo[0] += 0.9 * s[0] + 1.5 * s[1]
        hip_todo[1] += 0.9 * s[0] + 1.5 * s[1]

        # Adjust knee according to vertical speed, to damp bouncy oscillations
        knee_todo[0] -= 5.0 * s[3]      
        knee_todo[1] -= 5.0 * s[3]

        # Set action for next frame
        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5*a, -1.0, 1.0)

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
for t in range(2):
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
print('We ended up running like this. Parameters: Speed: %.2fx, %.2f, %.2f, %.2f, %.2f. Best best reward: %d' % (best_best_parameters[0], best_best_parameters[1], best_best_parameters[2], best_best_parameters[3], best_best_parameters[4], best_best_reward) )
reward = run_episode(env, best_best_parameters, True)

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
