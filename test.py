import time
import unittest
import pyglet
from pymarlzooplus import pymarlzooplus


class TestPymarlzooplusEnvironments(unittest.TestCase):

    pyglet.options['headless'] = True
    @classmethod
    def setUpClass(cls):
        # Set up parameters for all environments.
        cls.params_dict = {}

        # Arguments for PettingZoo
        cls.params_dict["PettingZoo"] = {
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
                "render_mode": "rgb_array",  # Options: "human", "rgb_array"
                "image_encoder": "ResNet18",  # Options: "ResNet18", "SlimSAM", "CLIP"
                "image_encoder_use_cuda": True,  # Whether to load image-encoder in GPU or not.
                "image_encoder_batch_size": 1,  # How many images to encode in a single pass.
                "partial_observation": False,
                "trainable_cnn": False,
                "kwargs": "",
                "seed": 2024
            }
        }

        # Arguments for Overcooked
        cls.params_dict["Overcooked"] = {
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
        cls.params_dict["PressurePlate"] = {
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
        cls.params_dict["LBF_v2"] = {
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
        cls.params_dict["LBF_v3"] = {
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
        cls.params_dict["RWARE_v1"] = {
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
        cls.params_dict["RWARE_v2"] = {
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
        cls.params_dict["MPE"] = {
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
        cls.params_dict["CaptureTarget"] = {
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
        cls.params_dict["BoxPushing"] = {
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

    def test_environments(self):
        completed = []
        failed = {}


        for name, params in self.params_dict.items():
            with self.subTest(environment=name):
                try:
                    pymarlzooplus(params)
                    completed.append(name)
                except (Exception, SystemExit) as e:
                    failed[name] = str(e)
                    self.fail(f"Test for '{name}' failed with exception: {e}")
                # Wait a short time to allow the experiment thread to exit.
                time.sleep(5)
        if failed:
            self.fail(f"Some tests failed: {failed}")

if __name__ == '__main__':
    unittest.main()
