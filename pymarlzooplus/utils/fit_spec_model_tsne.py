import os
import glob
import shutil
import subprocess
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.manifold import TSNE


def create_all_data_script(path_to_sacred, model_name_paths, test_nepisode):
    '''
    Script that will create a json file containing all the data from all the sacred runs for a given game
    Checks if saved models exist for a game
    '''
    sacred_directory = Path(path_to_sacred) / 'sacred'
    print(sacred_directory)

    model_name_paths = [Path(path) for path in model_name_paths]

    # Create the json files by running the appropriate command lines
    # Get config parameters
    # Creates a numbered subfolder, we remove it later on to avoid duplication
    for path in model_name_paths:
        config_path = path / "config.json"
        data = json.loads(config_path.read_text())
        alg_config = data["name"]
        env_config = data["env"]
        game_name = data["env_args"]["key"]
        max_steps = data["env_args"]["time_limit"]
        hidden_dim = data["hidden_dim"]

        # Run the evaluation -> creates a subfolder
        cmd_line = f"python main.py \
                    --config={alg_config} \
                    --env-config={env_config} \
                    with \
                    env_args.key={game_name} \
                    env_args.time_limit={max_steps} \
                    checkpoint_path={path} \
                    evaluate=True \
                    test_nepisode={test_nepisode} \
                    runner='episode'\
                    batch_size_run=1 \
                    hidden_dim={hidden_dim} \
                    "
        print('Creating new evaluation subfolders...')
        subprocess.run(cmd_line, shell=True)

    # get the paths of all evaluation subfolders (last folder created by sacred)
    # Flat: dict {episode_num: 'obs': [[obs1, obs2, ...], [obs1, obs2, ...], ...], 'actions': [[action1, action2, ...], [action1, action2, ...], ...]}
    # key_to_model_list: [model_name1, model_name2, ...] of length number of episodes
    flat, key_to_model_list, count = {}, [], 0

    for path in model_name_paths:
        run_parent_folder = path.parent  # should return the name of the game
        run_numbers=  [int(subf.parts[-1]) for subf in run_parent_folder.iterdir() if subf.parts[-1].isdigit()]
        all_data_folder = run_parent_folder / f"{max(run_numbers)}"

        # Get all the all_data.json files
        file_path = all_data_folder / "all_data.json"
        data = json.loads(file_path.read_text())

        for episode_key, episode_data in data.items():
            if episode_key == "action_meaning":
                action_mapping = episode_data
            else:
                ep_key = f"episode_{count}"
                flat[ep_key] = episode_data
                key_to_model_list.append(run_parent_folder.parts[-2])  # should return the name of the model
                count += 1
        # Delete the created subfolder once we are done
        if all_data_folder.exists():
            shutil.rmtree(all_data_folder)

    # Action key is a string, we need to convert it to an int
    action_mapping = {int(k):v for k,v in action_mapping.items()}

    return flat, key_to_model_list, action_mapping, game_name


def combine_data(flat, key_to_model_list):
    '''
    Given all the agent observations and actions, create list/array of all observations, action_states and actions
    Input:
    all_data: dict {episode_num: 'obs': [[obs1, obs2, ...], [obs1, obs2, ...], ...], 'actions': [[action1, action2, ...], [action1, action2, ...], ...]}
    
    Output:
    obs_list: list of all observations (steps x n_agents) x obs_size
    actions_list: list of all actions (steps x n_agents)
    states_list: list of all observation and state concatenation (step respective) (steps x n_agents) x (obs_size + obs_size)
    '''
    obs_list, actions_list, states_list, models_list = [], [], [], []
    for i, ep_value in enumerate(flat.values()):
        # steps x n_agents x obs_size
        ep_obs = np.array(ep_value["obs"], dtype=np.float32)
        # steps x n_agents
        ep_actions = np.array(ep_value["actions"], dtype=np.int32)

        steps, n_agents, obs_size = ep_obs.shape

        # (steps x n_agents) x obs_size
        obs_flat = ep_obs.reshape(steps * n_agents, obs_size)
        actions_flat = ep_actions.reshape(steps * n_agents)

        obs_list.append(obs_flat)
        actions_list.append(actions_flat)
        models_list += [key_to_model_list[i]] * steps * n_agents

        # Get mean of observations
        # steps x obs_size
        step_mean  = np.mean(ep_obs, axis=1)
        # We must recreate a array to concatenate with obs_flat
        # This means duplicating the step_mean every n_agents times
        # (steps x n_agents) x obs_size
        mean_expanded = np.repeat(step_mean[:, None, :], n_agents, axis=1)
        mean_flat = mean_expanded.reshape(steps * n_agents, obs_size)
        states_list.append(np.concatenate([obs_flat, mean_flat], axis=1))

    # Concatenate all episodes
    obs_list = np.concatenate(obs_list, axis=0)
    actions_list = np.concatenate(actions_list, axis=0)
    states_list = np.concatenate(states_list, axis=0)

    return obs_list, actions_list, states_list, models_list


def t_sne_to_dataframe(obs_list, actions_list, states_list, models_list, action_mapping):
    '''
    Projects the observations and state_observations to 2D using t-SNE algorithm
    Returns a dataframe with the following columns:
        - x_obs
        - y_obs
        - x_state
        - y_state
        - action
        - model_name
    with length equal to the total number of observations 
    (number of episodes * number of steps per episode * number of agents)
    '''
    print('Fitting t-SNE model...')
    ind_obs_embedded = TSNE(n_components=2, init='pca').fit_transform(obs_list)
    state_obs_embedded = TSNE(n_components=2, init='pca').fit_transform(states_list)

    dataframe = pd.DataFrame({
                    'x_obs':ind_obs_embedded[:,0],
                    'y_obs':ind_obs_embedded[:,1],
                    'x_state':state_obs_embedded[:,0],
                    'y_state':state_obs_embedded[:,1],
                    'action':actions_list,
                    'model_name':models_list
                    })
    
    dataframe['action_meaning'] = dataframe['action'].apply(lambda x: f"{x} : {action_mapping[x]}")
    # dataframe['action_meaning'] = dataframe['action'].apply(lambda x: f"{action_mapping[x]}")

    return dataframe

def plot(dataframe, path_to_save, game_name):
    # Distinct models
    model_names = sorted(list(set(dataframe['model_name'])))
    print(model_names)

    # Create folder if it doesn't exist
    path_to_save = path_to_save / game_name
    path_to_save.mkdir(parents=True, exist_ok=True)

    # Sort actions by ascending order for legend
    hue_levels = sorted(dataframe['action_meaning'].unique())
    palette_list = sns.color_palette("Set1", len(hue_levels))
    palette = dict(zip(hue_levels, palette_list))

    dataframe['action_meaning'] = pd.Categorical(dataframe['action_meaning'], categories=hue_levels, ordered=True)
    dataframe = dataframe.sort_values('action_meaning')

    for model_name in model_names:
        # Create folder for each model
        model_folder = path_to_save / model_name
        model_folder.mkdir(parents=True, exist_ok=True)

        model_df = dataframe[dataframe['model_name'] == model_name]
        print(f"Plotting model {model_name} with dataframe of shape {model_df.shape}")

        #----------------- t-SNE -----------------
        print('Plotting t-SNE...')
        # Plot observations to 2D
        plt.figure(figsize=(10,10))
        sns.scatterplot(x='x_obs', y='y_obs', hue='action_meaning', hue_order=hue_levels, data=model_df, 
                        legend='full', palette=palette, alpha=0.8)

        file_name = f'2d_t_sned_{model_name}'
        output_path = model_folder / f"{file_name}.png"
        plt.savefig(output_path, dpi=400)
        plt.close()

        # Plot states to 2D
        plt.figure(figsize=(10,10))
        sns.scatterplot(x='x_state', y='y_state', hue='action_meaning', hue_order=hue_levels, data=model_df, 
                        legend='full', palette=palette, alpha=0.8)
        
        file_name = f'2d_t_sne_observations_and_states_{model_name}'
        output_path = model_folder / f"{file_name}.png"
        plt.savefig(output_path, dpi=400)
        plt.close()

        #----------------- Action disparity -----------------
        print('Plotting action disparity...')
        action_disparity = pd.DataFrame(model_df['action_meaning'].value_counts())
        action_disparity['ratio'] = action_disparity['count'] / action_disparity['count'].sum()
        action_disparity = action_disparity.reindex(hue_levels)

        fig, ax = plt.subplots(figsize=(10,1.5))

        left = 0
        patches = []
        for action in hue_levels:
            ratio = action_disparity.loc[action, 'ratio']
            color = palette[action]
            ax.barh(y=0, left=left, width=ratio, height=0.5, color=color, edgecolor=None)
            patches.append(Patch(color=color, label=f"{action} ({ratio*100:.1f}%)"))
            left += ratio
        
        ax.axis('off')

        ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, 1.1), 
                  ncol=len(hue_levels), frameon=False)

        plt.tight_layout()

        file_name = f'action_disparity_{model_name}'
        output_path = model_folder / f"{file_name}.png"
        plt.savefig(output_path, dpi=100)
        plt.close()


def cli():
    p = argparse.ArgumentParser()
    default_dir = Path(os.path.dirname(os.path.abspath(__name__))) / 'results'
    
    p.add_argument("--sacred_directory", type=str, default=default_dir)
    p.add_argument("--model_paths", nargs='+', default=None) # ends with the run number and must contain 'models' folder
    p.add_argument("--test_nepisode", type=int, default=10)
    p.add_argument("--output_dir", type=str, default=None)

    args = p.parse_args()

    print(args.model_paths)

    # Create json files by running the appropriate command lines
    flat, key_to_model, action_mapping, game_name = create_all_data_script(args.sacred_directory, args.model_paths, args.test_nepisode)

    # Combine all episodes into one list
    obs_list, actions_list, states_list, models_list = combine_data(flat, key_to_model)

    # Fit t-SNE model and return dataframe for plotting
    dataframe = t_sne_to_dataframe(obs_list, actions_list, states_list, models_list, action_mapping)

    # Plot
    if args.output_dir is None:
        output_dir = default_dir / "tsne_plots"

    plot(dataframe, output_dir, game_name)

if __name__ == "__main__":
    cli()
    