from functools import partial

from smac.env import MultiAgentEnv
from envs.gym_wrapper import _GymmaWrapper
from envs.pettingzoo_wrapper import _PettingZooWrapper
from envs.overcooked_wrapper import _OvercookedWrapper
from envs.pressureplate_wrapper import _PressurePlateWrapper
import envs.lbf_registration
import envs.mpe_registration


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {
    "gymma": partial(env_fn, env=_GymmaWrapper),
    "pettingzoo": partial(env_fn, env=_PettingZooWrapper),
    "overcooked": partial(env_fn, env=_OvercookedWrapper),
    "pressureplate": partial(env_fn, env=_PressurePlateWrapper)
}

