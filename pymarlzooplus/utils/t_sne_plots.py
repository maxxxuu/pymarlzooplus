import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import json
import os
from pathlib import Path
from sklearn.manifold import TSNE
import pandas as pd

class TSNEPlotter:
    def __init__(self, all_data, action_mapping, path_to_save):
        self.all_data = all_data
        self.path_to_save = path_to_save
        self.action_mapping = action_mapping

    def data_transform(self, all_data):
        '''
        Given all the agent observations and actions, create list/array of all observations, action_states and actions
        Input:
        all_data: dict {episode_num: 'obs': [[obs1, obs2, ...], [obs1, obs2, ...], ...], 'actions': [[action1, action2, ...], [action1, action2, ...], ...]}
        
        Output:
        obs_list: list of all observations (steps x n_agents) x obs_size
        actions_list: list of all actions (steps x n_agents)
        states_list: list of all observation and state concatenation (step respective) (steps x n_agents) x (obs_size + obs_size)
        '''
        obs_list, actions_list, states_list = [], [], []
        for episode in all_data.values():
            # steps x n_agents x obs_size
            ep_obs = np.array(episode['obs'], dtype=np.float32)
            # steps x n_agents
            ep_actions = np.array(episode['actions'], dtype=np.int32)

            steps, n_agents, obs_size = ep_obs.shape

            # (steps x n_agents) x obs_size
            obs_flat = ep_obs.reshape(steps * n_agents, obs_size)
            actions_flat = ep_actions.reshape(steps * n_agents)

            obs_list.append(obs_flat)
            actions_list.append(actions_flat)

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

        # Turn to arrays
        obs_list = np.array(obs_list)
        actions_list = np.array(actions_list)
        states_list = np.array(states_list)

        return obs_list, actions_list, states_list
    
    def t_sne_to_dataframe(self, obs_list, actions_list, states_list, action_mapping):
        '''
        Projects the observations and state_observations to 2D using t-SNE algorithm
        Returns a dataframe with the following columns:
            - x_obs
            - y_obs
            - x_state
            - y_state
            - action
        with length equal to the total number of observations 
        (number of episodes * number of steps per episode * number of agents)
        '''
        print('Fitting t-SNE model...')
        ind_obs_embedded = TSNE(n_components=2, init='pca').fit_transform(obs_list)
        state_obs_embedded = TSNE(n_components=2, init='pcagi ').fit_transform(states_list)

        dataframe = pd.DataFrame({
                        'x_obs':ind_obs_embedded[:,0],
                        'y_obs':ind_obs_embedded[:,1],
                        'x_state':state_obs_embedded[:,0],
                        'y_state':state_obs_embedded[:,1],
                        'action':actions_list
                        })
        
        dataframe['action_meaning'] = dataframe['action'].apply(lambda x: f"{x} : {action_mapping[x]}")

        return dataframe
    
    def plot(self, dataframe, path_to_save):
        df = dataframe.copy()

        path_to_save.mkdir(parents=True, exist_ok=True)

        # Sort actions by ascending order for legend
        hue_levels = sorted(dataframe['action_meaning'].unique())
        palette_list = sns.color_palette("Set1", len(hue_levels))
        palette = dict(zip(hue_levels, palette_list))

        #----------------- t-SNE -----------------
        print('Plotting t-SNE...')
        # Plot observations to 2D
        plt.figure(figsize=(10,10))
        sns.scatterplot(x='x_obs', y='y_obs', hue='action_meaning', hue_order=hue_levels, data=df, 
                        legend='full', palette=palette, alpha=1)

        file_name = '2d_t_sned'
        output_path = path_to_save / f"{file_name}.png"
        plt.savefig(output_path, dpi=400)
        plt.close()

        # Plot states to 2D
        plt.figure(figsize=(10,10))
        sns.scatterplot(x='x_state', y='y_state', hue='action_meaning', hue_order=hue_levels, data=df, 
                        legend='full', palette=palette, alpha=1)
        
        file_name = '2d_t_sne_observations_and_states'
        output_path = path_to_save / f"{file_name}.png"
        plt.savefig(output_path, dpi=400)
        plt.close()

        #----------------- Action disparity -----------------
        action_disparity = pd.DataFrame(dataframe['action_meaning'].value_counts())
        action_disparity['ratio'] = action_disparity['count'] / action_disparity['count'].sum()
        action_disparity = action_disparity.reindex(hue_levels)

        figg, ax = plt.subplots(figsize=(10,1.5))

        left = 0
        patches = []
        for action in hue_levels:
            ratio = action_disparity.loc[action, 'ratio']
            color = palette[action]
            ax.barh(y=0, left=left, width=ratio, height=0.5, color=color, edgecolor=None)
            patches.append(Patch(color=color, label=action))
            left += ratio
        
        ax.axis('off')

        ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, 1.1), 
                  ncol=len(hue_levels), frameon=False)

        plt.tight_layout()

        file_name = 'action_disparity'
        output_path = path_to_save / f"{file_name}.png"
        plt.savefig(output_path, dpi=100)
        plt.close()

def plot_2d_t_sne(all_data, action_mapping, path_to_save):
    tsne_plotter = TSNEPlotter(all_data, action_mapping, path_to_save)
    # Transform data
    obs_list, actions_list, states_list = tsne_plotter.data_transform(all_data)
    # Create dataframe
    dataframe = tsne_plotter.t_sne_to_dataframe(obs_list, actions_list, states_list, action_mapping)
    # Plot
    tsne_plotter.plot(dataframe, path_to_save)
    print('t-SNE plots saved to', path_to_save)
