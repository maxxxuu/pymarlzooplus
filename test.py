import time

from pymarlzooplus import pymarlzooplus
from pymarlzooplus.utils.test_logger import Tee

params_dict = {}

# Arguments for PettingZoo
pz_params_dict = {
    "config": "qmix",
    "algo_args": {
        "test_nepisode": 2,
        "test_interval": 2,
        "t_max": 10,
        "log_interval": 2,
        "runner_log_interval": 2,
        "learner_log_interval": 2,
        "batch_size": 2,
        "buffer_size": 2,
    },
    "env-config": "pettingzoo",
    "env_args": {
        "key": "pistonball_v6",
        "time_limit": 2,  # Episode horizon.
        "render_mode": "rgb_array",  # Options: "human", "rgb_array
        "image_encoder": "ResNet18",  # Options: "ResNet18", "SlimSAM", "CLIP"
        "image_encoder_use_cuda": True,  # Whether to load image-encoder in GPU or not.
        "image_encoder_batch_size": 1,  # How many images to encode in a single pass.
        "partial_observation": False,  # Only for "Emtombed: Cooperative" and "Space Invaders"
        "trainable_cnn": False,  # Specifies whether to return image-observation or the encoded vector-observation
        "kwargs": "",
        "seed": 2024
    }
}
params_dict["PettingZoo"] = pz_params_dict

# Arguments for Overcooked
overcooked_params_dict = {
    "config": "qmix",
    "algo_args": {
        "test_nepisode": 2,
        "test_interval": 2,
        "t_max": 10,
        "log_interval": 2,
        "runner_log_interval": 2,
        "learner_log_interval": 2,
        "batch_size": 2,
        "buffer_size": 2,
    },
    "env-config": "overcooked",
    "env_args": {
        "time_limit": 2,
        "key": "coordination_ring",
        "reward_type": "sparse",
    }
}
params_dict['Overcooked'] = overcooked_params_dict

# Arguments for Pressure Plate
pressureplate_params_dict = {
    "config": "qmix",
    "algo_args": {
        "test_nepisode": 2,
        "test_interval": 2,
        "t_max": 10,
        "log_interval": 2,
        "runner_log_interval": 2,
        "learner_log_interval": 2,
        "batch_size": 2,
        "buffer_size": 2,
    },
    "env-config": "pressureplate",
    "env_args": {
        "key": "pressureplate-linear-4p-v0",
        "time_limit": 2,
        "seed": 2024
    }
}
params_dict['PressurePlate'] = pressureplate_params_dict

# Arguments for LBF version 2
lbfv2_params_dict = {
    "config": "qmix",
    "algo_args": {
        "test_nepisode": 2,
        "test_interval": 2,
        "t_max": 10,
        "log_interval": 2,
        "runner_log_interval": 2,
        "learner_log_interval": 2,
        "batch_size": 2,
        "buffer_size": 2,
    },
    "env-config": "gymma",
    "env_args": {
        "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
        "time_limit": 2,
        "seed": 2024
    }
}
params_dict['LBF_v2'] = lbfv2_params_dict

# Arguments for LBF version 3
lbfv3_params_dict = {
    "config": "qmix",
    "algo_args": {
        "test_nepisode": 2,
        "test_interval": 2,
        "t_max": 10,
        "log_interval": 2,
        "runner_log_interval": 2,
        "learner_log_interval": 2,
        "batch_size": 2,
        "buffer_size": 2,
    },
    "env-config": "gymma",
    "env_args": {
        "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v3",
        "time_limit": 2,
        "seed": 2024
    }
}
params_dict['LBF_v3'] = lbfv3_params_dict

# Arguments for RWARE version 1
rwarev1_params_dict = {
    "config": "qmix",
    "algo_args": {
        "test_nepisode": 2,
        "test_interval": 2,
        "t_max": 10,
        "log_interval": 2,
        "runner_log_interval": 2,
        "learner_log_interval": 2,
        "batch_size": 2,
        "buffer_size": 2,
    },
    "env-config": "gymma",
    "env_args": {
        "key": "rware:rware-small-4ag-hard-v1",
        "time_limit": 2,
        "seed": 2024
    }
}
params_dict['RWARE_v1'] = rwarev1_params_dict

# Arguments for RWARE version 2
rwarev2_params_dict = {
    "config": "qmix",
    "algo_args": {
        "test_nepisode": 2,
        "test_interval": 2,
        "t_max": 10,
        "log_interval": 2,
        "runner_log_interval": 2,
        "learner_log_interval": 2,
        "batch_size": 2,
        "buffer_size": 2,
    },
    "env-config": "gymma",
    "env_args": {
        "key": "rware:rware-small-4ag-hard-v2",
        "time_limit": 2,
        "seed": 2024
    }
}
params_dict['RWARE_v2'] = rwarev2_params_dict

# Arguments for MPE
mpe_params_dict = {
    "config": "qmix",
    "algo_args": {
        "test_nepisode": 2,
        "test_interval": 2,
        "t_max": 10,
        "log_interval": 2,
        "runner_log_interval": 2,
        "learner_log_interval": 2,
        "batch_size": 2,
        "buffer_size": 2,
    },
    "env-config": "gymma",
    "env_args": {
      "key": "mpe:SimpleSpeakerListener-v0",
      "time_limit": 2,
      "seed": 2024
    }
}
params_dict['MPE'] = mpe_params_dict

# Arguments for Capture Target
capturetarget_params_dict = {
    "config": "qmix",
    "algo_args": {
        "test_nepisode": 2,
        "test_interval": 2,
        "t_max": 10,
        "log_interval": 2,
        "runner_log_interval": 2,
        "learner_log_interval": 2,
        "batch_size": 2,
        "buffer_size": 2,
    },
    "env-config": "capturetarget",
    "env_args": {
        "time_limit": 2,
        "key": "CaptureTarget-6x6-1t-2a-v0",
        "seed": 2024
    }
}
params_dict['CaptureTarget'] = capturetarget_params_dict

# Arguments for Box Pushing
boxpushing_params_dict = {
    "config": "qmix",
    "algo_args": {
        "test_nepisode": 2,
        "test_interval": 2,
        "t_max": 10,
        "log_interval": 2,
        "runner_log_interval": 2,
        "learner_log_interval": 2,
        "batch_size": 2,
        "buffer_size": 2,
    },
    "env-config": "boxpushing",
    "env_args": {
        "key": "BoxPushing-6x6-2a-v0",
        "time_limit": 2,
        "seed": 2024
    }
}
params_dict['BoxPushing'] = boxpushing_params_dict

completed = []
failed = {}

# Open a log file to capture all outputs
with Tee("test_runs.log"):

    for k, v in params_dict.items():
        print("\n#############################################")
        print(f"Running test for {k} ...")
        try:
            pymarlzooplus(v)
            completed.append(k)
        except (Exception, SystemExit) as e:
            if isinstance(e, SystemExit):
                failed[k] = f"Sacred SystemExit (code {e.code}). \nPlease see the error printed above and the log file."
                print(f"\nTest '{k}' failed due to Sacred SystemExit (code {e.code})")
            else:
                failed[k] = str(e)
                print(f"\nTest '{k}' failed with exception: {e}")
        # Wait a little bit for the experiment thread to exit and stop
        time.sleep(5)

    if len(failed) == 0:
        print("\n#############################################")
        print("Test completed successfully!")
    else:
        print("\n\n#############################################")
        print("Tests completed with failures:")
        for k, v in failed.items():
            print("\n#############################################")
            print(f"{k}: \n{v}")
        print("\n\n#############################################")
        print("Tests completed successfully:")
        print(completed)
