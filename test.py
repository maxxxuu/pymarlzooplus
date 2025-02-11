from pymarlzooplus import pymarlzooplus

# params_dict = {
#     "config": "qmix",
#     "env-config": "capturetarget",
#     "env_args": {
#         "time_limit": 60,
#         "key" : "CaptureTarget-6x6-1t-2a-v0",
#     }
# }


params_dict = {
    "config": "qmix",
    "env-config": "overcooked",
    "env_args": {
        "time_limit": 500,
        "key": "coordination_ring",
        "reward_type": "sparse",
    }
}

pymarlzooplus(params_dict)
