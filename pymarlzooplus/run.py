import datetime
import os
from pathlib import Path
import pprint
import time
import threading
import torch as th
from tqdm import tqdm
from types import SimpleNamespace as SN
from pymarlzooplus.utils._logging import Logger
from pymarlzooplus.utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import importlib
import pickle
import json

from sacred.observers import FileStorageObserver  # This is for finding the results' path

from pymarlzooplus.learners import REGISTRY as le_REGISTRY
from pymarlzooplus.runners import REGISTRY as r_REGISTRY
from pymarlzooplus.controllers import REGISTRY as mac_REGISTRY
from pymarlzooplus.modules.explorers import REGISTRY as explorer_REGISTRY

from pymarlzooplus.components.episode_buffer import ReplayBuffer, PrioritizedReplayBuffer
from pymarlzooplus.components.transforms import OneHot
from pymarlzooplus.components.episodic_memory_buffer import EpisodicMemoryBuffer

from pymarlzooplus.utils.plot_utils import plot_single_experiment_results
from pymarlzooplus.utils.t_sne_plots import plot_2d_t_sne


def run(_run, _config, _log):

    # Find the file path where the results are stored in order to create the plots latter
    assert _run.observers, "No observers defined!"
    file_storage_observer_idx = -1
    for observer_idx, observer in enumerate(_run.observers):
        if isinstance(observer, FileStorageObserver):
            file_storage_observer_idx = observer_idx
            break  # Successfully identified the observer as FileStorageObserver
    assert file_storage_observer_idx > -1, "File storage observer is not defined!"
    results_dir = _run.observers[file_storage_observer_idx].dir
    

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    args.device_cnn_modules = "cuda" if args.use_cuda_cnn_modules else "cpu"
    
    args.results_dir = results_dir

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]
    unique_token = f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Wait a little bit before creating the plots to save the corresponding files
    time.sleep(10)

    # Plot results
    print("Creating plots ...")
    plot_single_experiment_results(results_dir, algo_name=_config['name'], env_name=map_name)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=2)
            if t.is_alive():
                print(f"Thread {t.name} did not terminate.")
            else:
                print(f"Thread {t.name} has joined successfully.")

    print("Exiting script")


def evaluate_sequential(args, runner):

    if args.evaluate == True and args.runner == 'episode':
        # Get action meanings
        env_copy = runner.env.original_env.unwrapped
        env_mod = importlib.import_module(env_copy.__class__.__module__)
        Action_Enum = getattr(env_mod, 'Action')
        action_to_meaning = {a.value: a.name for a in Action_Enum}

        all_data = {}
        # Run test episodes
        for _ in tqdm(range(args.test_nepisode)):
            batch, all_info = runner.run(test_mode=True)
            all_data[f'episode_{_}'] = all_info
        all_data['action_meaning'] = action_to_meaning

        # Save data as json
        path_to_save = os.path.join(args.results_dir, "all_data.json")
        with open(path_to_save, 'w') as f:
            json.dump(all_data, f)
        print('Saved state and action data to', path_to_save)

        # Create 2D t-SNE plots
        plot_output_dir = Path(args.results_dir) / 'tsne_plots'
        # plot_2d_t_sne(all_data, action_to_meaning, plot_output_dir, args.test_nepisode)

    else:
        for _ in range(args.test_nepisode):
            runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.unit_dim = env_info["obs_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    # Add extra filed in buffer
    if "log_probs" in args.extra_in_buffer:
        scheme["log_probs"] = {"vshape": (1,), "group": "agents"}
    if "values" in args.extra_in_buffer:
        scheme["values"] = {"vshape": (1,), "group": "agents"}
    if "hidden_states" in args.extra_in_buffer:
        if args.use_rnn is True:
            scheme["hidden_states"] = {"vshape": (args.hidden_dim,), "group": "agents"}
        else:
            args.extra_in_buffer = [item for item in args.extra_in_buffer if item != 'hidden_states']
    if "hidden_states_critic" in args.extra_in_buffer:
        if args.use_rnn_critic is True:
            scheme["hidden_states_critic"] = {"vshape": (args.hidden_dim,), "group": "agents"}
        else:
            args.extra_in_buffer = [item for item in args.extra_in_buffer if item != 'hidden_states_critic']

    # Define buffer
    if args.prioritized_buffer is True:
        buffer = PrioritizedReplayBuffer(
            scheme,
            groups,
            args.buffer_size,
            env_info["episode_limit"] + 1,
            args.prioritized_buffer_alpha,
            preprocess=preprocess,
            device="cpu" if args.buffer_cpu_only else args.device
        )
    else:
        buffer = ReplayBuffer(
            scheme,
            groups,
            args,
            args.buffer_size,
            env_info["episode_limit"] + 1,
            preprocess=preprocess,
            device="cpu" if args.buffer_cpu_only else args.device,
        )
    # Define episode buffer
    args.use_emdqn = getattr(args, "use_emdqn", False)
    if args.use_emdqn is True:
        ec_buffer = EpisodicMemoryBuffer(args, scheme)

    # Setup multi-agent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Setup the explorer, if applicable
    has_explorer = args.has_explorer
    explorer = None
    if args.has_explorer is True:
        explorer = explorer_REGISTRY[args.explorer](scheme, groups, args, env_info["episode_limit"])

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, explorer=explorer)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    mac.learner = learner

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directory {} doesn't exist".format(args.checkpoint_path)
            )
            return
        

        ########## ADDED FOR FIX ###########
        #  Get the model path
        model_path = os.path.join(args.checkpoint_path, 'models')

        # Load t_max from config
        with open(os.path.join(args.checkpoint_path, 'config.json'), 'r') as f:
            config = json.load(f)
            t_max = config['t_max']
        timestep_to_load= t_max

        # # Go through all files in args.checkpoint_path
        # for name in os.listdir(args.checkpoint_path):
        #     full_name = os.path.join(args.checkpoint_path, name)
        #     # Check if they are dirs the names of which are numbers
        #     if os.path.isdir(full_name) and name.isdigit():
        #         timesteps.append(int(name))

        # if args.load_step == 0:
        #     # choose the max timestep
        #     timestep_to_load = max(timesteps)
        # else:
        #     # choose the timestep closest to load_step
        #     timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        # model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            print('Evaluation mode')
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)

        # Update episode buffer
        if args.use_emdqn is True:
            ec_buffer.update_ec(episode_batch)

        # Update replay buffer
        buffer.insert_episode_batch(episode_batch)

        # Run training iterations
        for _ in range(args.num_circle):
            if buffer.can_sample(args.batch_size):
                if args.prioritized_buffer is True:
                    sample_indices, episode_sample = buffer.sample(args.batch_size)
                else:
                    episode_sample = buffer.sample(args.batch_size)

                # Train explorer. This is for EOI and CDS.
                if has_explorer is True:
                    explorer.train(episode_sample)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                # Learner training
                if args.prioritized_buffer is True:
                    if args.use_emdqn is True:
                        learner.train(episode_sample, runner.t_env, episode, ec_buffer=ec_buffer)
                    else:
                        td_error = learner.train(episode_sample, runner.t_env, episode)
                        buffer.update_priority(sample_indices, td_error)
                else:
                    if args.use_emdqn is True:
                        learner.train(episode_sample, runner.t_env, episode, ec_buffer=ec_buffer)
                    else:
                        learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        # if args.save_model and (
        #     runner.t_env - model_save_time >= args.save_model_interval
        #     or model_save_time == 0
        # ):
        #     model_save_time = runner.t_env
        #     save_path = os.path.join(
        #         args.local_results_path, "models", args.unique_token, str(runner.t_env)
        #     )
        #     os.makedirs(save_path, exist_ok=True)
        #     logger.console_logger.info("Saving models to {}".format(save_path))

        #     # learner should handle saving/loading -- delegate actor save/load to mac,
        #     # use appropriate filenames to do critics, optimizer states
        #     learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    # save model at the end of training
    if args.save_model:
        save_path = os.path.join(
            args.results_dir, "models"
        )
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving models to {}".format(save_path))
        # learner should handle saving/loading -- delegate actor save/load to mac,
        # use appropriate filenames to do critics, optimizer states
        learner.save_models(save_path)

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # Set CUDA flags
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )
    if config["use_cuda_cnn_modules"] and not th.cuda.is_available():
        config["use_cuda_cnn_modules"] = False
        _log.warning(
            "CUDA flag use_cuda_cnn_modules was switched OFF automatically because no CUDA devices are available!"
        )
    if config["use_cuda_cnn_modules"] is False and config["use_cuda"] is True:
        config["use_cuda_cnn_modules"] = True
        _log.warning(
            "'use_cuda_cnn_modules' was turned to True since 'use_cuda' is also True!"
        )

    # Set 'centralized_image_encoding' arg
    if "centralized_image_encoding" in list(config["env_args"].keys()):
        if config["env_args"]["centralized_image_encoding"] is True and config["batch_size_run"] == 1:
            config["env_args"]["centralized_image_encoding"] = False
            _log.warning(
                "'centralized_image_encoding' was turned to False since only 1 env process is running!"
            )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
