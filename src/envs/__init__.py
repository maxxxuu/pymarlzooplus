from functools import partial

from smac.env import MultiAgentEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY_availability = [
    "gymma",
    "pettingzoo",
    "overcooked",
    "pressureplate",
    "capturetarget",
    "boxpushing",
]

# In this way, the user don't need to install requirements
REGISTRY = {}

try:
    import envs.lbf_registration
except:
    pass

try:
    import envs.mpe_registration
except:
    pass

try:
    from envs.gym_wrapper import _GymmaWrapper
    REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)
except:
    pass

try:
  from envs.pettingzoo_wrapper import _PettingZooWrapper
  REGISTRY["pettingzoo"] = partial(env_fn, env=_PettingZooWrapper)
except:
    pass

try:
    from envs.overcooked_wrapper import _OvercookedWrapper
    REGISTRY["overcooked"] = partial(env_fn, env=_OvercookedWrapper)
except:
    pass

try:
    from envs.pressureplate_wrapper import _PressurePlateWrapper
    REGISTRY["pressureplate"] = partial(env_fn, env=_PressurePlateWrapper)
except:
    pass
  
try:
    from envs.capturetarget_wrapper import _CaptureTargetWrapper
    REGISTRY["capturetarget"] = partial(env_fn, env=_CaptureTargetWrapper)
except:
    pass

try:
    from envs.boxpushing_wrapper import _BoxPushingWrapper
    REGISTRY["boxpushing"] = partial(env_fn, env=_BoxPushingWrapper)
except:
    pass

