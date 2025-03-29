import time
import unittest
import traceback

import numpy as np

from pymarlzooplus import pymarlzooplus
from pymarlzooplus.envs import REGISTRY as env_REGISTRY


class TestPymarlzooplusEnvironments(unittest.TestCase):
    
    # noinspection PyUnresolvedReferences
    # noinspection PyDictCreation
    @classmethod
    def setUpClass(cls):
        # Set up parameters
        cls.train_framework_params_dict = {}
        cls.env_api_fully_coop_params_dict = {}
        cls.env_api_non_fully_coop_params_dict = {}

        ############################################################
        # Arguments to test training framework with PettingZoo with all algorithms

        pz_encoded_images_args = {
            "key": "pistonball_v6",
            "time_limit": 2,
            "render_mode": "human",
            "image_encoder": "ResNet18",
            "image_encoder_use_cuda": False,
            "image_encoder_batch_size": 1,
            "trainable_cnn": False,
            "kwargs": "",
            "seed": 2024
        }
        pz_raw_images_args = {
            "key": "pistonball_v6",
            "time_limit": 2,
            "render_mode": "human",
            "image_encoder": "ResNet18",
            "image_encoder_use_cuda": False,
            "image_encoder_batch_size": 1,
            "trainable_cnn": True,
            "kwargs": "",
            "seed": 2024
        }

        # Arguments for PettingZoo with QMIX
        cls.train_framework_params_dict["PettingZoo_QMIX"] = {
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
                "target_update_interval_or_tau": 4,
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with QMIX-NS
        cls.train_framework_params_dict["PettingZoo_QMIX-NS"] = {
            "config": "qmix-ns",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with CDS
        cls.train_framework_params_dict["PettingZoo_CDS"] = {
            "config": "cds",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval": 4
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with CDS
        cls.train_framework_params_dict["PettingZoo_CDS"] = {
            "config": "cds",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval": 4
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with COMA
        cls.train_framework_params_dict["PettingZoo_COMA"] = {
            "config": "coma",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with COMA-NS
        cls.train_framework_params_dict["PettingZoo_COMA-NS"] = {
            "config": "coma-ns",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with EMC
        cls.train_framework_params_dict["PettingZoo_EMC"] = {
            "config": "emc",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "burn_in_period": 2,
                "target_update_interval": 4
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with EOI
        cls.train_framework_params_dict["PettingZoo_EOI"] = {
            "config": "eoi",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with HAPPO
        cls.train_framework_params_dict["PettingZoo_HAPPO"] = {
            "config": "happo",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "data_chunk_length": 3,
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with IA2C
        cls.train_framework_params_dict["PettingZoo_IA2C"] = {
            "config": "ia2c",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with IA2C-NS
        cls.train_framework_params_dict["PettingZoo_IA2C-NS"] = {
            "config": "ia2c-ns",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with IPPO
        cls.train_framework_params_dict["PettingZoo_IPPO"] = {
            "config": "ippo",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with IPPO-NS
        cls.train_framework_params_dict["PettingZoo_IPPO-NS"] = {
            "config": "ippo-ns",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with IQL
        cls.train_framework_params_dict["PettingZoo_IQL"] = {
            "config": "iql",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval_or_tau": 4
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with IQL-NS
        cls.train_framework_params_dict["PettingZoo_IQL-NS"] = {
            "config": "iql-ns",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval_or_tau": 4
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with MAA2C
        cls.train_framework_params_dict["PettingZoo_MAA2C"] = {
            "config": "maa2c",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with MAA2C-NS
        cls.train_framework_params_dict["PettingZoo_MAA2C-NS"] = {
            "config": "maa2c-ns",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with MADDPG
        cls.train_framework_params_dict["PettingZoo_MADDPG"] = {
            "config": "maddpg",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval_or_tau": 4
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with MADDPG-NS
        cls.train_framework_params_dict["PettingZoo_MADDPG-NS"] = {
            "config": "maddpg-ns",
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
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with MAPPO
        cls.train_framework_params_dict["PettingZoo_MAPPO"] = {
            "config": "mappo",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with MAPPO-NS
        cls.train_framework_params_dict["PettingZoo_MAPPO-NS"] = {
            "config": "mappo-ns",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with MASER
        cls.train_framework_params_dict["PettingZoo_MASER"] = {
            "config": "maser",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval": 4
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with MAT-DEC
        cls.train_framework_params_dict["PettingZoo_MAT-DEC"] = {
            "config": "mat_dec",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with QPLEX
        cls.train_framework_params_dict["PettingZoo_QPLEX"] = {
            "config": "qplex",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval": 4
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with VDN
        cls.train_framework_params_dict["PettingZoo_VDN"] = {
            "config": "vdn",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval_or_tau": 4
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for PettingZoo with VDN-NS
        cls.train_framework_params_dict["PettingZoo_VDN-NS"] = {
            "config": "vdn-ns",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval_or_tau": 4
            },
            "env-config": "pettingzoo",
            "env_args": pz_encoded_images_args
        }

        # Arguments for Overcooked
        cls.train_framework_params_dict["Overcooked"] = {
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

        # Arguments for Pressure Plate
        cls.train_framework_params_dict["PressurePlate"] = {
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

        # Arguments for LBF version 2
        cls.train_framework_params_dict["LBF_v2"] = {
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

        # Arguments for LBF version 3
        cls.train_framework_params_dict["LBF_v3"] = {
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

        # Arguments for RWARE version 1
        cls.train_framework_params_dict["RWARE_v1"] = {
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

        # Arguments for RWARE version 2
        cls.train_framework_params_dict["RWARE_v2"] = {
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

        # Arguments for MPE
        cls.train_framework_params_dict["MPE"] = {
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

        # Arguments for Capture Target
        cls.train_framework_params_dict["CaptureTarget"] = {
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

        # Arguments for Box Pushing
        cls.train_framework_params_dict["BoxPushing"] = {
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

        ############################################################
        # Arguments to test training framework with LBF version 2 with all algorithms

        # Arguments for LBF version 2 with CDS
        cls.train_framework_params_dict["LBF_v2_CDS"] = {
            "config": "cds",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with COMA
        cls.train_framework_params_dict["LBF_v2_COMA"] = {
            "config": "coma",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with COMA-NS
        cls.train_framework_params_dict["LBF_v2_COMA-NS"] = {
            "config": "coma_ns",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with EMC
        cls.train_framework_params_dict["LBF_v2_EMC"] = {
            "config": "emc",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "burn_in_period": 2,
                "target_update_interval": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with EOI
        cls.train_framework_params_dict["LBF_v2_EOI"] = {
            "config": "eoi",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with HAPPO
        cls.train_framework_params_dict["LBF_v2_HAPPO"] = {
            "config": "happo",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "data_chunk_length": 3,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with IA2C
        cls.train_framework_params_dict["LBF_v2_IA2C"] = {
            "config": "ia2c",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with IA2C-NS
        cls.train_framework_params_dict["LBF_v2_IA2C-NS"] = {
            "config": "ia2c_ns",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with IPPO
        cls.train_framework_params_dict["LBF_v2_IPPO"] = {
            "config": "ippo",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with IPPO-NS
        cls.train_framework_params_dict["LBF_v2_IPPO-NS"] = {
            "config": "ippo_ns",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with IQL
        cls.train_framework_params_dict["LBF_v2_IQL"] = {
            "config": "iql",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with IQL-NS
        cls.train_framework_params_dict["LBF_v2_IQL-NS"] = {
            "config": "iql_ns",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with MAA2C
        cls.train_framework_params_dict["LBF_v2_MAA2C"] = {
            "config": "maa2c",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with MAA2C-NS
        cls.train_framework_params_dict["LBF_v2_MAA2C-NS"] = {
            "config": "maa2c_ns",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with MADDPG
        cls.train_framework_params_dict["LBF_v2_MADDPG"] = {
            "config": "maddpg",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with MADDPG-NS
        cls.train_framework_params_dict["LBF_v2_MADDPG-NS"] = {
            "config": "maddpg_ns",
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

        # Arguments for LBF version 2 with MAPPO
        cls.train_framework_params_dict["LBF_v2_MAPPO"] = {
            "config": "mappo",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with MAPPO-NS
        cls.train_framework_params_dict["LBF_v2_MAPPO-NS"] = {
            "config": "mappo_ns",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with MASER
        cls.train_framework_params_dict["LBF_v2_MASER"] = {
            "config": "maser",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with MAT-DEC
        cls.train_framework_params_dict["LBF_v2_MAT-DEC"] = {
            "config": "mat_dec",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "batch_size_run": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with QMIX-NS
        cls.train_framework_params_dict["LBF_v2_QMIX-NS"] = {
            "config": "qmix_ns",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with QPLEX
        cls.train_framework_params_dict["LBF_v2_QPLEX"] = {
            "config": "qplex",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with VDN
        cls.train_framework_params_dict["LBF_v2_VDN"] = {
            "config": "vdn",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        # Arguments for LBF version 2 with VDN-NS
        cls.train_framework_params_dict["LBF_v2_VDN-NS"] = {
            "config": "vdn_ns",
            "algo_args": {
                "test_nepisode": 2,
                "test_interval": 2,
                "t_max": 10,
                "log_interval": 2,
                "runner_log_interval": 2,
                "learner_log_interval": 2,
                "batch_size": 2,
                "buffer_size": 2,
                "target_update_interval_or_tau": 4,
            },
            "env-config": "gymma",
            "env_args": {
                "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
                "time_limit": 2,
                "seed": 2024
            }
        }

        ############################################################
        # Arguments to test environment API with all fully cooperative environments

        pettingzoo_fully_coop_env_keys = [
            "pistonball_v6",
            "cooperative_pong_v5",
            "entombed_cooperative_v3",
            "space_invaders_v2"
        ]
        for pettingzoo_fully_coop_env_key in pettingzoo_fully_coop_env_keys:
            cls.env_api_fully_coop_params_dict[pettingzoo_fully_coop_env_key] = {
                "env": "pettingzoo",
                "env_args": {
                    "key": pettingzoo_fully_coop_env_key,
                    "time_limit": 10,
                    "render_mode": "human",
                    "image_encoder": "ResNet18",
                    "image_encoder_use_cuda": False,
                    "image_encoder_batch_size": 2,
                    "trainable_cnn": False,
                    "kwargs": "",
                    "seed": 2024
                }
            }
            cls.env_api_fully_coop_params_dict[f"{pettingzoo_fully_coop_env_key}_raw_images"] = {
                "env": "pettingzoo",
                "env_args": {
                    "key": pettingzoo_fully_coop_env_key,
                    "time_limit": 10,
                    "render_mode": "human",
                    "image_encoder": "ResNet18",
                    "image_encoder_use_cuda": False,
                    "image_encoder_batch_size": 2,
                    "trainable_cnn": True,
                    "kwargs": "",
                    "seed": 2024
                }
            }

        pettingzoo_fully_coop_partial_obs_env_keys = [
            "entombed_cooperative_v3",
            "space_invaders_v2"
        ]
        for pettingzoo_fully_coop_partial_obs_env_key in pettingzoo_fully_coop_partial_obs_env_keys:
            cls.env_api_fully_coop_params_dict[f"{pettingzoo_fully_coop_partial_obs_env_key}_partial_observation"] = {
                "env": "pettingzoo",
                "env_args": {
                    "key": pettingzoo_fully_coop_partial_obs_env_key,
                    "time_limit": 10,
                    "render_mode": "human",
                    "image_encoder": "ResNet18",
                    "image_encoder_use_cuda": False,
                    "image_encoder_batch_size": 2,
                    "trainable_cnn": False,
                    "partial_observation": True,
                    "kwargs": "",
                    "seed": 2024
                }
            }
            cls.env_api_fully_coop_params_dict[
                f"{pettingzoo_fully_coop_partial_obs_env_key}_raw_images_partial_observation"
            ] = {
                "env": "pettingzoo",
                "env_args": {
                    "key": pettingzoo_fully_coop_partial_obs_env_key,
                    "time_limit": 10,
                    "render_mode": "human",
                    "image_encoder": "ResNet18",
                    "image_encoder_use_cuda": False,
                    "image_encoder_batch_size": 2,
                    "trainable_cnn": True,
                    "partial_observation": True,
                    "kwargs": "",
                    "seed": 2024
                }
            }

        overcooked_env_keys = [
            "coordination_ring",
            "asymmetric_advantages",
            "cramped_room",
            "counter_circuit",
            "random3",
            "random0",
            "unident",
            "forced_coordination",
            "soup_coordination",
            "small_corridor",
            "simple_tomato",
            "simple_o_t",
            "simple_o",
            "schelling_s",
            "schelling",
            "m_shaped_s",
            "long_cook_time",
            "large_room",
            "forced_coordination_tomato",
            "cramped_room_tomato",
            "cramped_room_o_3orders",
            "asymmetric_advantages_tomato",
            "bottleneck",
            "cramped_corridor",
            "counter_circuit_o_1order",
        ]
        for overcooked_env_key in overcooked_env_keys:
            cls.env_api_fully_coop_params_dict[f"{overcooked_env_key}_sparse"] = {
                "env": "overcooked",
                "env_args": {
                    "key": overcooked_env_key,
                    "time_limit": 10,
                    "reward_type": "sparse",
                    "seed": 2024
                }
            }
            cls.env_api_fully_coop_params_dict[f"{overcooked_env_key}_shaped"] = {
                "env": "overcooked",
                "env_args": {
                    "key": overcooked_env_key,
                    "time_limit": 10,
                    "reward_type": "shaped",
                    "seed": 2024
                }
            }

        pressureplate_env_keys = [
            "pressureplate-linear-4p-v0",
            "pressureplate-linear-5p-v0",
            "pressureplate-linear-6p-v0"
        ]
        for pressureplate_env_key in pressureplate_env_keys:
            cls.env_api_fully_coop_params_dict[pressureplate_env_key] = {
              "env": "pressureplate",
              "env_args": {
                  "key": pressureplate_env_key,
                  "time_limit": 10,
                  "seed": 2024
              }
            }

        lbf_env_keys = [
            "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
            "lbforaging:Foraging-2s-11x11-3p-2f-coop-v2",
            "lbforaging:Foraging-2s-8x8-3p-2f-coop-v2",
            "lbforaging:Foraging-2s-9x9-3p-2f-coop-v2",
            "lbforaging:Foraging-7s-20x20-5p-3f-coop-v2",
            "lbforaging:Foraging-2s-12x12-2p-2f-coop-v2",
            "lbforaging:Foraging-8s-25x25-8p-5f-coop-v2",
            "lbforaging:Foraging-7s-30x30-7p-4f-coop-v2",
            "lbforaging:Foraging-4s-11x11-3p-2f-coop-v3",
            "lbforaging:Foraging-2s-11x11-3p-2f-coop-v3",
            "lbforaging:Foraging-2s-8x8-3p-2f-coop-v3",
            "lbforaging:Foraging-2s-9x9-3p-2f-coop-v3",
            "lbforaging:Foraging-7s-20x20-5p-3f-coop-v3",
            "lbforaging:Foraging-2s-12x12-2p-2f-coop-v3",
            "lbforaging:Foraging-8s-25x25-8p-5f-coop-v3",
            "lbforaging:Foraging-7s-30x30-7p-4f-coop-v3",
        ]
        for lbf_env_key in lbf_env_keys:
            cls.env_api_fully_coop_params_dict[lbf_env_key] = {
                "env": "gymma",
                "env_args": {
                    "key": lbf_env_key,
                    "time_limit": 10,
                    "seed": 2024,
                }
            }

        rware_env_keys = [
            "rware:rware-small-4ag-hard-v1",
            "rware:rware-tiny-4ag-hard-v1",
            "rware:rware-tiny-2ag-hard-v1",
            "rware:rware-small-4ag-hard-v2",
            "rware:rware-tiny-4ag-hard-v2",
            "rware:rware-tiny-2ag-hard-v2",
        ]
        for rware_env_key in rware_env_keys:
            cls.env_api_fully_coop_params_dict[rware_env_key] = {
              "env": "gymma",
              "env_args": {
                  "key": rware_env_key,
                  "time_limit": 10,
                  "seed": 2024
              }
            }

        mpe_env_keys = [
            "mpe:SimpleSpeakerListener-v0",
            "mpe:SimpleSpread-3-v0",
            "mpe:SimpleSpread-4-v0",
            "mpe:SimpleSpread-5-v0",
            "mpe:SimpleSpread-8-v0",
            "mpe:MultiSpeakerListener-v0",
            "mpe:SimpleAdversary-v0",
            "mpe:SimpleCrypto-v0",
            "mpe:SimplePush-v0",
            "mpe:SimpleReference-v0",
            "mpe:SimpleTag-v0",
            "mpe:SimpleWorldComm-v0",
            "mpe:ClimbingSpread-v0",
        ]
        for mpe_env_key in mpe_env_keys:
            cls.env_api_fully_coop_params_dict[mpe_env_key] = {
                "env": "gymma",
                "env_args": {
                    "key": mpe_env_key,
                    "time_limit": 10,
                    "seed": 2024
                }
            }

        capturetarget_env_keys = ["CaptureTarget-6x6-1t-2a-v0"]
        for capturetarget_env_key in capturetarget_env_keys:
            cls.env_api_fully_coop_params_dict[capturetarget_env_key] = {
                "env": "capturetarget",
                "env_args": {
                    "key": capturetarget_env_key,
                    "time_limit": 10,
                    "seed": 2024
                }
            }
            cls.env_api_fully_coop_params_dict[f"{capturetarget_env_key}_obs_one_hot"] = {
                "env": "capturetarget",
                "env_args": {
                    "key": capturetarget_env_key,
                    "time_limit": 10,
                    "seed": 2024,
                    "obs_one_hot": True
                }
            }
            cls.env_api_fully_coop_params_dict[f"{capturetarget_env_key}_wo_tgt_avoid_agent"] = {
                "env": "capturetarget",
                "env_args": {
                    "key": capturetarget_env_key,
                    "time_limit": 10,
                    "seed": 2024,
                    "tgt_avoid_agent": False
                }
            }

        boxpushing_env_keys = ["BoxPushing-6x6-2a-v0"]
        for boxpushing_env_key in boxpushing_env_keys:
            cls.env_api_fully_coop_params_dict[boxpushing_env_key] = {
                "env": "boxpushing",
                "env_args": {
                    "key": boxpushing_env_key,
                    "time_limit": 10,
                    "seed": 2024
                }
            }

        ###########################################################
        # Arguments to test environment API with all NON-fully cooperative PettingZoo environments (except for Classic)
        pettingzoo_non_fully_coop_env_atari_keys = [
            "basketball_pong_v3",
            "boxing_v2",
            "combat_plane_v2",
            "combat_tank_v2",
            "double_dunk_v3",
            "entombed_competitive_v3",
            "flag_capture_v2",
            "foozpong_v3",
            "ice_hockey_v2",
            "joust_v3",
            "mario_bros_v3",
            "maze_craze_v3",
            "othello_v3",
            "pong_v3",
            "quadrapong_v4",
            "space_war_v2",
            "surround_v2",
            "tennis_v3",
            "video_checkers_v4",
            "volleyball_pong_v3",
            "warlords_v3",
            "wizard_of_wor_v3"
        ]
        for pettingzoo_non_fully_coop_env_atari_key in pettingzoo_non_fully_coop_env_atari_keys:
            cls.env_api_non_fully_coop_params_dict[pettingzoo_non_fully_coop_env_atari_key] = {
                "env": "pettingzoo",
                "env_args": {
                    "key": pettingzoo_non_fully_coop_env_atari_key,
                    "time_limit": 10,
                    "render_mode": "human",
                    "image_encoder": "ResNet18",
                    "image_encoder_use_cuda": False,
                    "image_encoder_batch_size": 2,
                    "trainable_cnn": False,
                    "kwargs": "",
                    "seed": 2024
                }
            }
            cls.env_api_non_fully_coop_params_dict[f"{pettingzoo_non_fully_coop_env_atari_key}_raw_images"] = {
                "env": "pettingzoo",
                "env_args": {
                    "key": pettingzoo_non_fully_coop_env_atari_key,
                    "time_limit": 10,
                    "render_mode": "human",
                    "image_encoder": "ResNet18",
                    "image_encoder_use_cuda": False,
                    "image_encoder_batch_size": 2,
                    "trainable_cnn": True,
                    "kwargs": "",
                    "seed": 2024
                }
            }

        pettingzoo_non_fully_coop_env_butterfly_mpe_sisl_keys = [
            "knights_archers_zombies_v10",
            "simple_v3",
            "simple_adversary_v3",
            "simple_crypto_v3",
            "simple_push_v3",
            "simple_reference_v3",
            "simple_speaker_listener_v4",
            "simple_spread_v3",
            "simple_tag_v3",
            "simple_world_comm_v3",
            "multiwalker_v9",
            "pursuit_v4",
            "waterworld_v4",
        ]
        for pettingzoo_non_fully_coop_env_butterfly_mpe_sisl_key in (
                pettingzoo_non_fully_coop_env_butterfly_mpe_sisl_keys
        ):
            cls.env_api_non_fully_coop_params_dict[pettingzoo_non_fully_coop_env_butterfly_mpe_sisl_key] = {
                "env": "pettingzoo",
                "env_args": {
                    "key": pettingzoo_non_fully_coop_env_butterfly_mpe_sisl_key,
                    "time_limit": 10,
                    "render_mode": "human",
                    "kwargs": "",
                    "seed": 2024
                }
            }

        ###########################################################
        # Arguments to test environment API with all PettingZoo Classic environments
        cls.env_api_pz_classic_params_dict = {}
        pettingzoo_classic_keys = [
            "chess_v6",
            "connect_four_v3",
            "gin_rummy_v4",
            "go_v5",
            "hanabi_v5",
            "leduc_holdem_v4",
            "rps_v2",
            "texas_holdem_no_limit_v6",
            "texas_holdem_v4",
            "tictactoe_v3",
        ]
        for pettingzoo_classic_key in pettingzoo_classic_keys:
            cls.env_api_pz_classic_params_dict[pettingzoo_classic_key] = {
                "env": "pettingzoo",
                "env_args": {
                    "key": pettingzoo_classic_key,
                    "render_mode": "human",
                    "kwargs": "",
                }
            }

    def test_training_framework(self):
        completed = []
        failed = {}

        for name, params in self.train_framework_params_dict.items():
            print(
                "\n\n###########################################"
                "\n###########################################"
                f"\nRunning test for: {name}\n"
            )
            with self.subTest(environment=name):
                try:
                    pymarlzooplus(params)
                    completed.append(name)
                except (Exception, SystemExit) as e:
                    # Convert the full traceback to a string
                    tb_str = traceback.format_exc()
                    # Store the traceback
                    failed[name] = tb_str
                    # Print just a simple message
                    self.fail(f"Test for '{name}' failed with exception: {e}")
                # Wait a short time to allow the experiment thread to terminate.
                time.sleep(5)
        if failed:
            # Build a multiline message that unittest will print in full
            msg = "Some tests failed:\n"
            for name, tb_str in failed.items():
                msg += f"\n=== {name} ===\n{tb_str}\n"
            self.fail(msg)

    # def test_env_api_fully_coop(self):
    #     completed = []
    #     failed = {}
    #
    #     for name, params in self.env_api_fully_coop_params_dict.items():
    #         print(
    #             "\n\n###########################################"
    #             "\n###########################################"
    #             f"\nRunning test for: {name}\n"
    #         )
    #         with self.subTest(environment=name):
    #             try:
    #
    #                 # Initialize environment
    #                 env = env_REGISTRY[params["env"]](**params["env_args"])
    #
    #                 n_agns = env.get_n_agents()
    #                 assert isinstance(n_agns, int)
    #
    #                 n_acts = env.get_total_actions()
    #                 assert isinstance(n_acts, int)
    #
    #                 # Reset the environment
    #                 obs, state = env.reset()
    #                 assert isinstance(obs, tuple)
    #                 assert len(obs) == n_agns
    #                 obs_shape = obs[0].shape
    #                 for agent_obs in obs:
    #                     assert isinstance(agent_obs, np.ndarray)
    #                     assert agent_obs.shape == obs_shape
    #                 assert isinstance(state, np.ndarray)
    #                 if len(state.shape) == 1:
    #                     assert len(obs_shape) == 1
    #                     assert state.shape[0] == obs_shape[0] * n_agns
    #                 else:
    #                     assert len(state.shape) == 4
    #                     assert len(obs_shape) == 3
    #                     assert state.shape == (n_agns, *obs_shape)
    #
    #                 done = False
    #                 # Run an episode
    #                 while not done:
    #                     # Render the environment (optional)
    #                     env.render()
    #
    #                     # Insert the policy's actions here
    #                     actions = env.sample_actions()
    #                     assert isinstance(actions, list)
    #                     assert len(actions) == n_agns
    #                     for action in actions:
    #                         assert isinstance(action, int)
    #
    #                     # Apply an environment step
    #                     reward, done, extra_info = env.step(actions)
    #                     assert isinstance(reward, float)
    #                     assert isinstance(done, bool)
    #                     assert isinstance(extra_info, dict)
    #                     assert len(extra_info) == 0
    #
    #                     info = env.get_info()
    #                     assert 'TimeLimit.truncated' in list(info.keys())
    #                     assert isinstance(info['TimeLimit.truncated'], bool)
    #
    #                     obs = env.get_obs()
    #                     state = env.get_state()
    #                     assert isinstance(obs, tuple)
    #                     assert len(obs) == n_agns
    #                     obs_shape = obs[0].shape
    #                     for agent_obs in obs:
    #                         assert isinstance(agent_obs, np.ndarray)
    #                         assert agent_obs.shape == obs_shape
    #                     assert isinstance(state, np.ndarray)
    #                     if len(state.shape) == 1:
    #                         assert len(obs_shape) == 1
    #                         assert state.shape[0] == obs_shape[0] * n_agns
    #                     else:
    #                         assert len(state.shape) == 4
    #                         assert len(obs_shape) == 3
    #                         assert state.shape == (n_agns, *obs_shape)
    #
    #                 # Terminate the environment
    #                 env.close()
    #
    #                 completed.append(name)
    #             except (Exception, SystemExit) as e:
    #                 # Convert the full traceback to a string
    #                 tb_str = traceback.format_exc()
    #                 # Store the traceback
    #                 failed[name] = tb_str
    #                 # Print just a simple message
    #                 self.fail(f"Test for '{name}' failed with exception: {e}")
    #             # Wait a short time to allow the environment to terminate.
    #             time.sleep(2)
    #     if failed:
    #         # Build a multiline message that unittest will print in full
    #         msg = "Some tests failed:\n"
    #         for name, tb_str in failed.items():
    #             msg += f"\n=== {name} ===\n{tb_str}\n"
    #         self.fail(msg)

    # def test_env_api_non_fully_coop(self):
    #     completed = []
    #     failed = {}
    #
    #     for name, params in self.env_api_non_fully_coop_params_dict.items():
    #         print(
    #             "\n\n###########################################"
    #             "\n###########################################"
    #             f"\nRunning test for: {name}\n"
    #         )
    #         with self.subTest(environment=name):
    #             try:
    #
    #                 # Initialize environment
    #                 env = env_REGISTRY[params["env"]](**params["env_args"])
    #
    #                 n_agns = env.get_n_agents()
    #                 assert isinstance(n_agns, int)
    #
    #                 common_observation_space = env.common_observation_space()
    #                 assert isinstance(common_observation_space, bool)
    #
    #                 is_image = env.is_image()
    #                 assert isinstance(is_image, bool)
    #
    #                 agent_prefix = env.get_agent_prefix()
    #                 assert isinstance(agent_prefix, list)
    #                 assert len(agent_prefix) == n_agns
    #                 for _agent_prefix in agent_prefix:
    #                     assert isinstance(_agent_prefix, str)
    #
    #                 # Reset the environment
    #                 _obs, _state = env.reset()
    #
    #                 def check_obs_state(obs, state):
    #                     # Check the type of obs, its shape and
    #                     # (in case of common observation space) if all agents have the same observation space
    #                     assert isinstance(obs, dict)
    #                     assert len(obs) == n_agns
    #                     obs_shape = list(obs.values())[0].shape
    #                     for (agent_id, agent_obs), agent_prefix_id in zip(obs.items(), agent_prefix):
    #                         assert agent_id == agent_prefix_id
    #                         assert isinstance(agent_obs, np.ndarray)
    #                         if common_observation_space is True:
    #                             assert agent_obs.shape == obs_shape
    #                             if is_image is True:
    #                                 assert (
    #                                         (len(agent_obs.shape) == 3 and agent_obs.shape[0] == 3) or  # raw images
    #                                         len(agent_obs.shape) == 1  # encoded images
    #                                 )
    #                             else:
    #                                 assert len(agent_obs.shape) == 1, "agent_obs.shape: {}".format(agent_obs.shape)
    #                         else:
    #                             assert is_image is False
    #                             assert len(agent_obs.shape) == 1, "agent_obs.shape: {}".format(agent_obs.shape)
    #
    #                     # Check the type of state and its shape
    #                     if common_observation_space is True:
    #                         assert isinstance(state, np.ndarray)
    #                         if is_image is True:
    #                             if len(state.shape) == 4:  # raw images
    #                                 assert state.shape == (n_agns, *obs_shape)
    #                             else:  # encoded images
    #                                 assert len(state.shape) == 1
    #                                 assert len(obs_shape) == 1
    #                                 assert state.shape == (n_agns * obs_shape[0],)
    #                         else:
    #                             assert len(state.shape) == 2
    #                             assert state.shape == (n_agns, obs_shape[0])
    #                     else:
    #                         assert isinstance(state, np.ndarray)
    #                         assert len(state.shape) == 1
    #                         state_shape = 0
    #                         for agent_obs in obs.values():
    #                             assert len(agent_obs.shape) == 1
    #                             state_shape += agent_obs.shape[0]
    #                         assert state.shape[0] == state_shape
    #
    #                 check_obs_state(_obs, _state)
    #
    #                 done = False
    #                 # Run an episode
    #                 while not done:
    #
    #                     # Render the environment (optional)
    #                     env.render()
    #
    #                     # Insert the policy's actions here
    #                     actions = env.sample_actions()
    #                     # Check the actions
    #                     assert isinstance(actions, list)
    #                     assert len(actions) == n_agns
    #                     for action in actions:
    #                         assert (
    #                             isinstance(action, int) or
    #                             (isinstance(action, np.ndarray) and env.key in ["waterworld_v4", "multiwalker_v9"])
    #                         )
    #
    #                     # Apply an environment step
    #                     reward, done, info = env.step(actions)
    #                     # Check the rewards
    #                     assert isinstance(reward, dict)
    #                     assert len(reward) == n_agns
    #                     for (_agent_id, _reward), _agent_prefix_id in zip(reward.items(), agent_prefix):
    #                         assert _agent_id == _agent_prefix_id
    #                         assert isinstance(
    #                             _reward, (np.int64, int, np.float64, float)
    #                         ), "type(_reward): {}".format(type(_reward))
    #                     # Check the dones
    #                     assert isinstance(done, dict)
    #                     assert len(done) == n_agns
    #                     for (_agent_id, _done), _agent_prefix_id in zip(done.items(), agent_prefix):
    #                         assert _agent_id == _agent_prefix_id
    #                         assert isinstance(_done, bool)
    #                     # Check the infos
    #                     assert isinstance(info, dict)
    #                     assert 'TimeLimit.truncated' in list(info.keys())
    #                     assert isinstance(info['TimeLimit.truncated'], bool)
    #                     assert 'infos' in list(info.keys())
    #                     assert isinstance(info['infos'], dict)
    #                     assert 'truncations' in list(info.keys())
    #                     assert isinstance(info['truncations'], dict)
    #                     for (_agent_id, _), _agent_prefix_id in zip(info['infos'].items(), agent_prefix):
    #                         assert _agent_id == _agent_prefix_id
    #                     for (_agent_id, _truncation), _agent_prefix_id in zip(
    #                             info['truncations'].items(), agent_prefix
    #                     ):
    #                         assert _agent_id == _agent_prefix_id
    #                         assert isinstance(_truncation, bool)
    #                     assert info['TimeLimit.truncated'] == any(
    #                         [_truncation for _truncation in info['truncations'].values()]
    #                     )
    #
    #                     done = all([agent_done for agent_done in done.values()])
    #                     _obs = env.get_obs()
    #                     _state = env.get_state()
    #                     check_obs_state(_obs, _state)
    #
    #                 # Terminate the environment
    #                 env.close()
    #
    #                 completed.append(name)
    #             except (Exception, SystemExit) as e:
    #                 # Convert the full traceback to a string
    #                 tb_str = traceback.format_exc()
    #                 # Store the traceback
    #                 failed[name] = tb_str
    #                 # Print just a simple message
    #                 self.fail(f"Test for '{name}' failed with exception: {e}")
    #             # Wait a short time to allow the environment to terminate.
    #             time.sleep(2)
    #     if failed:
    #         # Build a multiline message that unittest will print in full
    #         msg = "Some tests failed:\n"
    #         for name, tb_str in failed.items():
    #             msg += f"\n=== {name} ===\n{tb_str}\n"
    #         self.fail(msg)

    # def test_env_api_pz_classic(self):
    #     completed = []
    #     failed = {}
    #
    #     for name, params in self.env_api_pz_classic_params_dict.items():
    #         print(
    #             "\n\n###########################################"
    #             "\n###########################################"
    #             f"\nRunning test for: {name}\n"
    #         )
    #         with self.subTest(environment=name):
    #             try:
    #
    #                 # Initialize environment
    #                 env = env_REGISTRY[params["env"]](**params["env_args"]).original_env
    #
    #                 # Get agents prefix
    #                 agents_prefix = [agent_prefix for agent_prefix in env.possible_agents]
    #                 n_agnts = len(agents_prefix)
    #
    #                 # Reset environment
    #                 env.reset(seed=42)
    #
    #                 # Run only for 10 steps, just for testing the functionality
    #                 count_steps = 0
    #
    #                 # Run an episode
    #                 for agent in env.agent_iter():
    #
    #                     # Render the environment (optional)
    #                     env.render()
    #
    #                     # Get environment data
    #                     observation, reward, termination, truncation, info = env.last()
    #                     # Check obs
    #                     if name != "rps_v2":  # Except 'Rock Paper Scissors'
    #                         assert isinstance(observation, dict)
    #                         assert ['observation', 'action_mask'] == list(observation.keys())
    #                         assert isinstance(observation['observation'], np.ndarray)
    #                         assert isinstance(observation['action_mask'], np.ndarray)
    #                     else:
    #                         assert isinstance(observation, np.ndarray), f"type(observation): {type(observation)}"
    #                     # Check reward
    #                     assert isinstance(reward, (int, np.int64, float)), f"type(reward): {type(reward)}"
    #                     # Check termination
    #                     assert isinstance(termination, bool)
    #                     # Check truncation
    #                     assert isinstance(truncation, bool)
    #                     # Check info
    #                     assert isinstance(info, dict)
    #                     if name == "hanabi_v5":
    #                         assert ['action_mask'] == list(info.keys())
    #                         assert isinstance(info['action_mask'], np.ndarray)
    #                         assert np.all(info['action_mask'] == observation['action_mask'])
    #                     else:
    #                         assert len(info) == 0
    #
    #                     # Get action
    #                     if termination or truncation:
    #                         action = None
    #                     else:
    #                         if name != "rps_v2":  # Except 'Rock Paper Scissors'
    #                             mask = observation["action_mask"]
    #                         # this is where you would insert your policy
    #                         if name != "rps_v2":  # Except 'Rock Paper Scissors'
    #                             action = env.action_space(agent).sample(mask)
    #                         else:
    #                             action = env.action_space(agent).sample()
    #                         # Check action
    #                         assert isinstance(action, np.int64)
    #
    #                     # Apply an environment step
    #                     env.step(action)
    #                     # Stop after 10 steps
    #                     count_steps += 1
    #                     if count_steps >= 10:
    #                         break
    #
    #                 # Terminate the environment
    #                 env.close()
    #
    #                 completed.append(name)
    #             except (Exception, SystemExit) as e:
    #                 # Convert the full traceback to a string
    #                 tb_str = traceback.format_exc()
    #                 # Store the traceback
    #                 failed[name] = tb_str
    #                 # Print just a simple message
    #                 self.fail(f"Test for '{name}' failed with exception: {e}")
    #             # Wait a short time to allow the environment to terminate.
    #             time.sleep(2)
    #     if failed:
    #         # Build a multiline message that unittest will print in full
    #         msg = "Some tests failed:\n"
    #         for name, tb_str in failed.items():
    #             msg += f"\n=== {name} ===\n{tb_str}\n"
    #         self.fail(msg)


if __name__ == '__main__':
    unittest.main()
