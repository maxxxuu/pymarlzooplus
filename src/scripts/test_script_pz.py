##################### entombed_cooperative_v3 #####################
from pettingzoo.atari import entombed_cooperative_v3
import cv2

env = entombed_cooperative_v3.parallel_env(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

    for observation in observations.values():
        image = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", image)
        cv2.waitKey(0)

env.close()
#########################################################################

##################### entombed_cooperative_v3 #####################
# from pettingzoo.atari import space_invaders_v2
# import cv2
#
# env = space_invaders_v2.parallel_env(render_mode="human")
# observations, infos = env.reset()
#
# while env.agents:
#     # this is where you would insert your policy
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#
#     observations, rewards, terminations, truncations, infos = env.step(actions)
#
#     for observation in observations.values():
#         image = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
#         cv2.imshow("image", image)
#         cv2.waitKey(0)
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
# import cv2
# from pettingzoo.butterfly import pistonball_v6
#
# env = pistonball_v6.parallel_env(render_mode="human", continuous=True, n_pistons=20, max_cycles=20)
#
# for iter_ in range(10):
#
#     env.reset()
#
#     counter = 0
#     while env.agents:
#         # this is where you would insert your policy
#         actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#
#         observations, rewards, terminations, truncations, infos = env.step(actions)
#         # image = env.render()
#         for observation in observations.values():
#             image = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
#             cv2.imshow("image", image)
#             cv2.waitKey(0)
#
#         counter += 1
#         print(counter)
#
# env.close()
#########################################################################
