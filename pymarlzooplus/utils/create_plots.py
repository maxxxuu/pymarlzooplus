import argparse
import json, os
from pathlib import Path
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import re


# Each run saves a directory that includes METRICS and CONFIG:
# METRICS contains the metrics.json file
# CONFIG contains the config.json file

# ------------------------------------------------------------------------------
#                           Data parsing functions
# ------------------------------------------------------------------------------

def info_parser(sacred_directory):
    games = set()       # List of all games found
    models = set()      # List of all models found
    game_groups = {}    # Groups game tasks by base game name
    model_game = {}    # Groups all games by model
    game_model = {}    # Groups all models that were trained for each game
    for dir in sacred_directory.iterdir():      # model dimension
        if not dir.is_dir():
            continue
        # All folders will be appended to model_game (including non model folders)
        model_game[dir.name] = set()
        for sub_dir in dir.iterdir():           # game dimension
            if not sub_dir.is_dir():
                continue
            sub_dir_name = sub_dir.name
            if sub_dir_name.split('-')[-1] == 'v1':  # Assumes all games end with v1
                games.add(sub_dir_name)
                # Add game to group
                base_game_name = sub_dir_name.split('-')[0]
                if base_game_name not in game_groups:
                    game_groups[base_game_name] = set()

                # Add game to model game
                if sub_dir_name not in game_model:
                    game_model[sub_dir_name] = set()
                
                # Add values
                game_groups[base_game_name].add(sub_dir_name)
                models.add(dir.name)
                model_game[dir.name].add(sub_dir_name)
                game_model[sub_dir_name].add(dir.name)

    return games, game_groups, models, model_game, game_model

def results_parser(sacred_directory, models):
    sacred_directory = Path(sacred_directory)

    # Get all games names that end with v1 (specifying game name annoying lolol) and also groups them
    # Could be improved in the games_parser function
    if models is None:
        game_names, game_groups, models, model_game, game_model = info_parser(sacred_directory)
    else:
        game_names, game_groups, _, model_game, game_model = info_parser(sacred_directory)

    print('Games:', game_names)

    metric_names = ['return_mean', 'return_std', 'test_return_mean', 'test_return_std']

    dataframes = {}

    # Create dataframes for each metric and game
    # One column per run (model_name + run_number)
    for game in game_names:
        dataframes[game] = {}
        for metric in metric_names:
            dfs = [] # Store dataframes for each model then concatenate
            for model in models:
                model_dir = sacred_directory / model
                if not model_dir.is_dir():
                    continue
                # find all experiment dirs that contain this game name
                for exp_dir in model_dir.iterdir():
                    if game not in exp_dir.name or not exp_dir.is_dir():
                        continue
                    for run_dir in exp_dir.iterdir():
                        # Check if numbered directory if not skip
                        if not run_dir.name.isdigit() or not run_dir.is_dir():
                            continue
                        # Check if metrics.json exists
                        metrics_path = run_dir / "metrics.json"
                        if not metrics_path.is_file():
                            print(f"{metrics_path} not found: SKIPPED")
                            continue
                        # Load data frmo json
                        with open(metrics_path) as f:
                            metrics = json.load(f)

                        # Check if metrics is empty
                        try:
                            metrics[metric]['values']
                        except KeyError:
                            print(f"No values for {metric} in {metrics_path}")

                        steps = metrics[metric]['steps']
                        values = metrics[metric]['values']

                        col_name = model + '_' + run_dir.name
                        s = pd.Series(data=values,
                                      index=steps,
                                      name=col_name)
                        dfs.append(s)
            # Concatenate all dataframes for this game, columns= one per run
            if dfs:
                dataframes[game][metric] = pd.concat(dfs, axis=1)
            else:
                print(f"No runs for found for any model for game={game}, metric={metric}")
                dataframes[game][metric] = pd.DataFrame()

    # Get best run for each game
    # Last value of test_return_mean is the best
    best_model = {}
    for game in game_names:
        best_model[game] = {}
        for model in models:
            best_value = float('-inf')
            best_run = None
            # Get runs for this model
            runs = [column_name for column_name in dataframes[game]['test_return_mean'].columns if model == column_name.rsplit('_', 1)[0]]
            for run in runs:
                value = dataframes[game]['test_return_mean'][run].iloc[-1]
                if value > best_value:
                    best_value = value
                    best_run = run.split('_')[-1]
            best_model[game][model] = best_run

    return dataframes, best_model, game_names, game_groups, model_game, game_model

# Get data for a specific model variant given a game name
def get_model_variant_data(dataframes, game_name, model_variant):
    df_wide_train = dataframes[game_name]['return_mean'].copy()
    df_wide_test = dataframes[game_name]['test_return_mean'].copy()

    assert df_wide_train.columns.tolist() == df_wide_test.columns.tolist()

    models_to_plot = [col_name for col_name in df_wide_train.columns.tolist() 
                      if col_name.rsplit('_', 1)[0] == model_variant]

    df_wide_train = df_wide_train[models_to_plot]
    df_wide_test = df_wide_test[models_to_plot]

    return df_wide_train, df_wide_test

def prepare_avg_data_for_plotting(dataframes, game_name, game_model, ci_z=1.15): 
    #1.15 is z-value for 0.75 confidence interval
    models_to_plot = game_model[game_name]
    avg_data = {}

    for model in models_to_plot:
        avg_data[model] = {}
        df_wide = get_model_variant_data(dataframes, game_name, model)
        # returns [df_wide_train, df_wide_test]
        for df, typee in zip(df_wide, ['train', 'test']):
            steps = df.index.to_numpy()
            mean = df.mean(axis=1).to_numpy()
            std = df.std(axis=1).to_numpy()
            n_samples = df.shape[0]
            if n_samples > 1:
                sem = std / np.sqrt(n_samples)
                ci = ci_z * sem
                ci_lower = mean - ci
                ci_upper = mean + ci
            else:
                ci_lower = ci_upper = np.zeros_like(mean)
            avg_data[model][typee + '_steps'] = steps
            avg_data[model][typee + '_mean'] = mean
            avg_data[model][typee + '_lower'] = ci_lower
            avg_data[model][typee + '_upper'] = ci_upper
    return avg_data

def prepare_best_data_for_plotting(dataframes, game_name, best_model):
    models_to_plot = {model:run_num for model, run_num in best_model[game_name].items() if run_num is not None}
    best_data = {}

    for model, run_num in models_to_plot.items():
        best_data[model] = {}
        for typee in ['', 'test_']:
            df_mean = dataframes[game_name][typee+'return_mean']
            df_std = dataframes[game_name][typee+'return_std']

            best_data[model][typee+'steps'] = df_mean.index.to_numpy()
            best_data[model][typee+'mean'] = df_mean[model+'_'+run_num].to_numpy()
            std = df_std[model+'_'+run_num].to_numpy()
            best_data[model][typee+'lower'] = best_data[model][typee+'mean'] - std
            best_data[model][typee+'upper'] = best_data[model][typee+'mean'] + std
    return best_data



# ------------------------------------------------------------------------------
#                              Plotting functions
# ------------------------------------------------------------------------------

def palette_choice(models, test_only):
    """
    Returns a dict {model: [colors]}:
      - test_only == False : 2 colors per model (train+test) with Paired colors
      - test_only == True  : 1 color per model (test only) from HLS colors (wheel)
    """
    models = list(models)
    if test_only:                      # one colour per model
        palette = sns.color_palette("hls", n_colors=len(models))
        model_color = {m: [c] for m, c in zip(models, palette)}
    else:                              # two colours per model
        palette = sns.color_palette("Paired", n_colors=len(models) * 2)
        model_color = {}
        for i, m in enumerate(models):
            model_color[m] = [palette[2 * i], palette[2 * i + 1]]
    return model_color

def common_legend(all_models, palette, test_only):
    legend_handles = []
    for model in all_models:
        linestyle = '--' if model == model.split('_')[0] else '-'
        colors = palette[model]
        if test_only:
            handle = Line2D([0], [0], color=colors[0], linestyle=linestyle, label=model)
            legend_handles.append(handle)
        else:
            handle_train = Line2D([0], [0], color=colors[0], linestyle=linestyle, label=model+'_train')
            handle_test = Line2D([0], [0], color=colors[1], linestyle=linestyle, label=model+'_test')
            legend_handles.extend([handle_train, handle_test])
    return legend_handles

def create_plots(data, game_name, best_or_avg, output_dir, palette=None, test_only=False, fill_between=True, ax=None):
    '''
    Plot data convention
    data[model] = {
        'train_steps': array,
        'train_mean':  array,
        'train_lower': array,
        'train_upper': array,
        'test_steps':  array,
        'test_mean':   array,
        'test_lower':  array,
        'test_upper':  array,
      }
    '''
    models_to_plot = list(data.keys())

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        new_fig = True
    else:
        new_fig = False

    sns.set_theme(style="whitegrid", context="talk")
    
    if palette is None:  # Indiv figure
        model_color = palette_choice(models_to_plot, test_only)
    else:                # Subplot figure
        model_color = palette

    for model in models_to_plot:
        # Dotted line for the base model (according to naming convention e.g. MAPPO_ + cent_pe)
        linestyle = 'dotted' if model == model.split('_')[0] else 'solid'
        # Train
        if not test_only:
            to_plot = ['train', 'test']
            labels = [model + '_train', model + '_test']
        else:
            to_plot = ['test']
            labels = [model]

        for i, (typee, label) in enumerate(zip(to_plot, labels)):
            col = model_color[model][i]
            x_data = data[model][typee+'_steps']
            y_data = data[model][typee+'_mean']
            sns.lineplot(x=x_data, y=y_data, label=label, color=col, linestyle=linestyle, ax=ax)
            if fill_between:
                lower = data[model][typee+'_lower']
                upper = data[model][typee+'_upper']
                ax.fill_between(x_data, lower, upper, color=col, alpha=0.15)

    # Only save for new figure (individual plots)
    if new_fig:
        ax.set_xlabel('Training steps')
        ax.set_ylabel('Reward')
        game_title = re.sub(r'[\ue000-\uf8ff]', '-', game_name)
        ax.set_title(f'{best_or_avg} model reward evolution for {game_title}')
        ax.legend(loc='lower right')
        filename = f"{best_or_avg}_model_{game_name}" + ("_test_only" if test_only else "")
        out_path = build_output(best_or_avg, False, output_dir, filename, test_only)

        plt.tight_layout()
        plt.savefig(out_path, dpi=400)
        plt.close()
    else:
        game_title = re.sub(r'[\ue000-\uf8ff]', '-', game_name)
        ax.set_title(game_title)

def create_subplots(base_game, same_game, dataframes, all_models, model_type, best_or_avg, output_dir, test_only=False, fill_between=True):
    '''
    model_type: best or avg
    same_game: list of games/tasks that belong to the same base game
    '''
    n_tasks = len(same_game)
    fig, axs = plt.subplots(1, n_tasks, figsize=(10*n_tasks, 10), sharex=True)
    if n_tasks ==1:
        axs = [axs]

    game_title = re.sub(r'[\ue000-\uf8ff]', '-', base_game)
    fig.suptitle(f"{best_or_avg} model reward evolution for tasks of {game_title}")

    if best_or_avg == 'best':
        plot_function = prepare_best_data_for_plotting
    elif best_or_avg == 'avg':
        plot_function = prepare_avg_data_for_plotting

    # Get palette
    palette = palette_choice(all_models, test_only)
    legend_handles = common_legend(all_models, palette, test_only)

    for ax, game_name in zip(axs, same_game):
        plot_data = plot_function(dataframes, game_name, model_type)
        create_plots(plot_data, game_name, best_or_avg, output_dir=None, palette=palette, test_only=test_only, fill_between=fill_between, ax=ax)
        ax.get_legend().remove()

    # Set axis labels
    mid = len(axs)//2
    axs[mid].set_xlabel("Training steps", labelpad=20)
    axs[0].set_ylabel('Episodic Reward', labelpad=20)

    # Common legend for all subplots
    fig.legend(handles=legend_handles, loc='lower center', ncol=11, frameon=False, 
               fontsize='large', handletextpad=0.5, columnspacing=1, bbox_to_anchor=(0.5, 0.0))
    
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])

    # Save figure
    filename = f"{best_or_avg}_model_{base_game}_subtasks" + ("_test_only" if test_only else "")
    out_path = build_output(best_or_avg, True, output_dir, filename, test_only)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


# ------------------------------------------------------------------------------
#                               MAIN
# ------------------------------------------------------------------------------

def build_output(best_or_avg, is_subplot, output_dir, filename, test_only):
    '''
    Returns a pathlib.Path where the plot will be saved
    3 subdirectories are meant to be created:
    - subplots: for tasks of the same base game
      - best_subplots: for best models subplots
      - avg_subplots: for avg models subplots
    - best_models: for best models plots
    - avg_models: for avg models plots
    '''
    root = Path(output_dir) if output_dir else Path('plots')

    test_or_train = 'test_only' if test_only else 'train_and_test'

    if is_subplot:
        folder = root / 'subplots'  / f"{best_or_avg}_subplots" / test_or_train
    else:
        folder = root / f"{best_or_avg}_models" / test_or_train

    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{filename}.png"

def cli():
    p = argparse.ArgumentParser()
    # Directory conventions
    p.add_argument("--sacred_directory", type=str, default="./")
    p.add_argument("--models", nargs='+', default=None)
    p.add_argument("--game", type=str, required=None)
    p.add_argument("--test_only", action="store_true")
    p.add_argument("--fill_between", action="store_false")
    p.add_argument("--linestyle", type=str, default='solid')
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--output_file_name", type=str, default=None)
    p.add_argument("--normalized", action="store_true")
    args = p.parse_args()

    return args


if __name__ == "__main__":
    args = cli()

    dataframes, best_model, game_names, game_groups, model_game, game_model = results_parser(args.sacred_directory, args.models)

    print('Plotting standalone games')

    for game_name in game_names:
        # Best plots
        best_data = prepare_best_data_for_plotting(dataframes, game_name, best_model)
        create_plots(best_data, game_name, 'best', args.output_dir, None, args.test_only, args.fill_between)

        # Average plots
        avg_data = prepare_avg_data_for_plotting(dataframes, game_name, game_model)
        create_plots(avg_data, game_name, 'avg', args.output_dir, None, args.test_only, args.fill_between)

    # Get models for each base game
    base_game_model = {}
    for game, subgames in game_groups.items():
        base_game_model[game] = set()
        for subgame in subgames:
            base_game_model[game].update(list(game_model[subgame]))

    print('Plotting subplots of games')

    for base_game in game_groups.keys():
        same_game = list(game_groups[base_game])

        all_models = list(base_game_model[base_game])

        # Best plots
        create_subplots(base_game, same_game, dataframes, all_models, best_model, 'best', args.output_dir, args.test_only, args.fill_between)

        # Avg plots
        create_subplots(base_game, same_game, dataframes, all_models, game_model, 'avg', args.output_dir, args.test_only, args.fill_between)
