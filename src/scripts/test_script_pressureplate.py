from pressureplate.environment import PressurePlate
import gym
import cv2
import random

env = gym.make("pressureplate-linear-4p-v0")

print(f"action_space: {env.action_space}")
print(f"observation_space: {env.observation_space}")

n_episodes = 10
num_agents = 4
for i in range(n_episodes):
    print(f"Episodes: {i}")

    obs = env.reset()

    done = [False] * num_agents
    counter = 0
    while not all(done):

        # Render
        image = env.render()
        print(env.action_space)
        print("and dim \n")
        print(env.action_space_dim)
        print("\n")
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow("Pressureplate", image)
        cv2.waitKey(500)

        # Fix the order of actions, always 'policy_agent_idx' corresponds to agent 0
        _actions = [random.randint(0, 3) for a in range(env.n_agents)]
        actions = _actions.copy()
        print(actions)
        obs, reward, done, info = env.step(actions)

        # print(f"done: {done}")
        # print(f"reward: {reward}")
        # print(f"info: {info}")

        counter += 1
        print(f"\nstep: {counter}, rew {reward}")

env.close()
