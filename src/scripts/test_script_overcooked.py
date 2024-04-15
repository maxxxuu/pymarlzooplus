from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import gym
import cv2
import random

mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
base_env = OvercookedEnv.from_mdp(mdp, horizon=50)
env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

print(f"action_space: {env.action_space}")
print(f"observation_space: {env.observation_space}")

n_episodes = 2
for i in range(n_episodes):
    print(f"Episodes: {i}")

    obs = env.reset()
    other_agent_idx = obs['other_agent_env_idx']
    agent_policy_idx = 1 - other_agent_idx

    done = False
    counter = 0
    while not done:

        # Render
        image = env.render()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow("Overcooked", image)
        cv2.waitKey(1)

        # Fix the order of actions, always 'policy_agent_idx' corresponds to agent 0
        _actions = [random.randint(0, 1), 0]
        if agent_policy_idx == 0:
            actions = _actions.copy()
        else:
            actions = _actions[::-1]  # reverse the order
        obs, reward, done, info = env.step(actions)
        # reward: single scalar value
        # done: single boolean
        # The provided reward is sum of the sparse rewards of both agents
        # (where the individual rewards are the same for both of them)
        # We can calculate also the sum the shaped rewards:
        shaped_reward = sum(info['shaped_r_by_agent'])
        print(f"shaped_reward: {shaped_reward}")

        # print(f"done: {done}")
        # print(f"reward: {reward}")
        # print(f"info: {info}")

        counter += 1
        print(f"\nstep: {counter}")
        # print(f"obs: {obs}")
        # print(f"info: {info}")

        # Fix the order of observations, always 'policy_agent_idx' corresponds to agent 0
        assert agent_policy_idx == info['policy_agent_idx']
        assert other_agent_idx == obs['other_agent_env_idx']
        _obs = [obs['both_agent_obs'][agent_policy_idx], obs['both_agent_obs'][other_agent_idx]]

env.close()
