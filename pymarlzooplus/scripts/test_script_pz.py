import copy

import numpy as np


def replace_color(image_, target_color, replacement_color):
    # Find all pixels matching the target color
    matches = np.all(image_ == target_color, axis=-1)
    # Replace these pixels with the replacement color
    image_[matches] = replacement_color

    return image_


##################### entombed_cooperative_v3 #####################
def entombed_cooperative_v3_get_combined_images(image_a, image_b):
    # Define the RGB values of agent 1
    agent_1_rgb_values = np.array([232, 232, 74])
    # Define the RGB values of agent 2
    agent_2_rgb_values = np.array([197, 124, 238])
    # Find where image A has the specific RGB values of agent 1
    mask = np.all(image_a == agent_1_rgb_values, axis=-1)
    # Find where image B has the specific RGB value of agent 1
    mask_ = np.all(image_b == agent_1_rgb_values, axis=-1)
    # Find which is the image which illustrates agent 1
    if mask_.sum() > mask.sum():
        print("Images order is wrong! Changing them ...")
        image_b = image_a
        mask = mask_
    # Replace the corresponding values in image B where the mask is True (that is where agent 1 is located)
    combined_image = image_b.copy()
    combined_image[mask] = agent_1_rgb_values
    # Create partial obs of agent 1 by removing agent 2 from the combined image
    black_rgb_values = [0, 0, 0]
    agent_1_obs_ = replace_color(combined_image.copy(), agent_1_rgb_values, black_rgb_values)
    # Create partial obe of agent 2 by removing agent 1 from the combined image
    agent_2_obs_ = replace_color(combined_image.copy(), agent_2_rgb_values, black_rgb_values)

    return combined_image, agent_1_obs_, agent_2_obs_


# from pettingzoo.atari import entombed_cooperative_v3
# import cv2
#
# env = entombed_cooperative_v3.parallel_env(render_mode="human")
#
# # Reset the environment
# previous_observations, previous_infos = env.reset()
# previous_obs = list(previous_observations.values())[0]
#
# # Perform no action in order to sync obs and actions
# no_actions = {'first_0': 0, 'second_0': 0}
# observations, rewards, terminations, truncations, infos = env.step(no_actions)
# current_obs = list(observations.values())[1]
#
# # Get the first obs of agents
# combined_obs, agent_1_obs, agent_2_obs = entombed_cooperative_v3_get_combined_images(previous_obs, current_obs)
#
# n_steps = 0
# while env.agents:
#     # this is where you would insert your policy
#
#     actions = {'first_0': 4, 'second_0': 3} # {agent: env.action_space(agent).sample() for agent in env.agents} #  {'first_0': np.random.choice([5, 1]), 'second_0': np.random.choice([5, 1])}   # {'first_0': 3, 'second_0': 4} # Right, Left  # {agent: 5 for agent in env.agents}
#
#     # In this step, agent 1 is moving
#     previous_observations_, previous_rewards_, previous_terminations_, previous_truncations_, previous_infos_ = env.step(actions)
#     previous_obs_ = list(previous_observations.values())[0]
#
#     # In this step, agent 2 is moving
#     observations_, rewards_, terminations_, truncations_, infos_ = env.step(actions)
#     current_obs_ = list(observations_.values())[1]
#
#     # Perform no action 2 times in order to sync obs and actions
#     previous_observations, previous_rewards, previous_terminations, previous_truncations, previous_infos = env.step(no_actions)
#     previous_obs = list(previous_observations.values())[0]
#     observations, rewards, terminations, truncations, infos = env.step(no_actions)
#     current_obs = list(observations.values())[1]
#     combined_obs, agent_1_obs, agent_2_obs = entombed_cooperative_v3_get_combined_images(previous_obs, current_obs)
#
#     print(n_steps)
#     print(actions)
#
#     # Render the state and observations
#     cv2.imshow("previous_obs_", cv2.cvtColor(previous_obs_, cv2.COLOR_RGB2BGR))
#     cv2.imshow("current_obs_", cv2.cvtColor(current_obs_, cv2.COLOR_RGB2BGR))
#     cv2.imshow("previous_obs", cv2.cvtColor(previous_obs, cv2.COLOR_RGB2BGR))
#     cv2.imshow("current_obs", cv2.cvtColor(current_obs, cv2.COLOR_RGB2BGR))
#     cv2.imshow("agent 1 obs", cv2.cvtColor(agent_1_obs, cv2.COLOR_RGB2BGR))
#     cv2.imshow("agent 2 obs", cv2.cvtColor(agent_2_obs, cv2.COLOR_RGB2BGR))
#     cv2.imshow("combined image", cv2.cvtColor(combined_obs, cv2.COLOR_RGB2BGR))
#     cv2.waitKey(0)
#
#     # for obs_agent_idx, observation in enumerate(observations.values()):
#     #     print(f"obs_agent_idx: {obs_agent_idx}")
#     #     image = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
#     #     cv2.imshow("image", image)
#     #     cv2.waitKey(0)
#
# env.close()
#########################################################################


##################### space_invaders_v2 #####################
def space_invaders_v2_get_combined_images(image_a, image_b, sensitivity=0):
    # We should remove the red ship, and the two agents from image A in order to get
    # just their final position from image B, otherwise artifacts will be created due
    # to the minor movements of these objects.
    red_ship_rgb_values = [181, 83, 40]
    agent_1_rgb_values = [50, 132, 50]
    agent_2_rgb_values = [162, 134, 56]
    image_a = replace_color(image_a, red_ship_rgb_values, [0, 0, 0])
    image_a = replace_color(image_a, agent_1_rgb_values, [0, 0, 0])
    image_a = replace_color(image_a, agent_2_rgb_values, [0, 0, 0])
    # Calculate the absolute difference between images
    diff = cv2.absdiff(image_a, image_b)
    # Convert the difference to grayscale in order to handle single threshold for all channels
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    # Mask for common areas: where the difference is less than or equal to sensitivity
    common_mask = np.where(diff_gray <= sensitivity, 255, 0).astype(np.uint8)
    # Mask for differences: where the difference is greater than sensitivity
    difference_mask = np.where(diff_gray > sensitivity, 255, 0).astype(np.uint8)
    # Create a 3-channel mask for common and difference areas
    common_mask_3channel = cv2.cvtColor(common_mask, cv2.COLOR_GRAY2RGB)
    difference_mask_3channel = cv2.cvtColor(difference_mask, cv2.COLOR_GRAY2RGB)
    # Extract common areas using common mask
    common_areas = cv2.bitwise_and(image_a, common_mask_3channel)
    # Extract differences from both images
    differences_from_a = cv2.bitwise_and(image_a, difference_mask_3channel)
    differences_from_b = cv2.bitwise_and(image_b, difference_mask_3channel)
    # Combine common areas with differences from both images
    combined_image = cv2.add(common_areas, differences_from_a)
    combined_image = cv2.add(combined_image, differences_from_b)
    # Create partial obs of agent 1 by removing agent 2 from the combined image
    agent_1_obs_ = replace_color(combined_image.copy(), agent_2_rgb_values, [0, 0, 0])
    # Create partial obe of agent 2 by removing agent 1 from the combined image
    agent_2_obs_ = replace_color(combined_image.copy(), agent_1_rgb_values, [0, 0, 0])

    return combined_image, agent_1_obs_, agent_2_obs_


# from pettingzoo.atari import space_invaders_v2
# import cv2
#
# env = space_invaders_v2.parallel_env(render_mode="human", max_cycles=10000)
# previous_observations, previous_infos = env.reset()
# previous_obs = list(previous_observations.values())[0]
#
# # Perform no action in order to sync obs and actions
# no_actions = {'first_0': 0, 'second_0': 0}
# observations, rewards, terminations, truncations, infos = env.step(no_actions)
# current_obs = list(observations.values())[1]
#
# # Get the first obs of agents
# combined_obs, agent_1_obs, agent_2_obs = space_invaders_v2_get_combined_images(previous_obs, current_obs)
#
# n_steps = 0
# while env.agents:
#     # this is where you would insert your policy
#     actions = {'first_0': np.random.choice([1, 4]), 'second_0': np.random.choice([1, 3])}  # {'first_0': 4, 'second_0': 3} # Right, Left # {agent: env.action_space(agent).sample() for agent in env.agents}
#     move_actions = {'first_0': actions['first_0'] if actions['first_0'] != 1 else 0,
#                     'second_0': actions['second_0'] if actions['second_0'] != 1 else 0
#                     }
#     fire_actions = {'first_0': actions['first_0'] if actions['first_0'] == 1 else 0,
#                     'second_0': actions['second_0'] if actions['second_0'] == 1 else 0
#                     }
#
#     # Perform the decided actions in order to get the state which is not full due to flickering
#     previous_observations, previous_rewards, previous_terminations, previous_truncations, previous_infos = env.step(move_actions)
#     previous_obs = list(previous_observations.values())[0]
#
#     # Perform no action and get the next state which is not full due to flickering
#     observations, rewards, terminations, truncations, infos = env.step(fire_actions)
#     current_obs = list(observations.values())[1]
#
#     # Combine the two states to get the full state after applying the actions
#     combined_obs, agent_1_obs, agent_2_obs = space_invaders_v2_get_combined_images(previous_obs, current_obs)
#
#     n_steps += 2
#
#     print(n_steps)
#     print(actions)
#
#     # Render the state and observations
#     cv2.imshow("image A", cv2.cvtColor(previous_obs, cv2.COLOR_RGB2BGR))
#     cv2.imshow("image B", cv2.cvtColor(current_obs, cv2.COLOR_RGB2BGR))
#     cv2.imshow("agent 1 obs", cv2.cvtColor(agent_1_obs, cv2.COLOR_RGB2BGR))
#     cv2.imshow("agent 2 obs", cv2.cvtColor(agent_2_obs, cv2.COLOR_RGB2BGR))
#     cv2.imshow("combined image", cv2.cvtColor(combined_obs, cv2.COLOR_RGB2BGR))
#     cv2.waitKey(0)
#
#     # for observation in observations.values():
#     #     image = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
#     #     cv2.imshow("image", image)
#     #     cv2.waitKey(0)
#
# env.close()
#########################################################################

##################### cooperative_pong_v5 #####################
import cv2
from pettingzoo.butterfly import cooperative_pong_v5

# env = cooperative_pong_v5.parallel_env(render_mode="rgb_array")
# # print(env.possible_agents)
# # exit()
#
# observations, infos = env.reset()
#
# while env.agents:
#     # this is where you would insert your policy
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#
#     observations, rewards, terminations, truncations, infos = env.step(actions)
#     for observation in observations.values():
#         image = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
#         cv2.imshow("image", image)
#         cv2.waitKey(0)
#
# env.close()
#########################################################################

##################### pistonball_v6 #####################
import cv2
from pettingzoo.butterfly import pistonball_v6

env = pistonball_v6.parallel_env(render_mode="human", continuous=True, n_pistons=20, max_cycles=20)

for iter_ in range(10):

    env.reset()

    counter = 0
    while env.agents:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
        print()
        print(rewards)
        print(infos)
        # image = env.render()
        # for observation in observations.values():
        #     image = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        #     cv2.imshow("image", image)
        #     cv2.waitKey(0)

        counter += 1
        print(counter)

env.close()
#########################################################################
