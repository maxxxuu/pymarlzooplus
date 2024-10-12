from gym.envs.registration import register

register(
    id="BoxPushing-v0",
    entry_point="box_pushing_ai_py.environment:BoxPushing",
)
