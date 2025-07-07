import argparse
import json, os
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns


# Each run saves a directory that includes METRICS and CONFIG:
# METRICS contains the metrics.json file
# CONFIG contains the config.json file

def get_metrics_config(directory):
    # Get the metrics.json file
    with open(directory + "/metrics.json") as f:
        metrics = json.load(f)

    # Get the config.json file
    with open(directory + "/config.json") as f:
        config = json.load(f)
    return metrics, config

def results_parser(sacred_directory, models, game_name):
    base_dir = sacred_directory

    metric_list = ['return_mean', 'return_std', 'test_return_mean', 'test_return_std']

    # Create data dictionary
    # data[model][game][metric]
    data = {}
    for model in models:
        data[model]            = {}
        data[model]['average'] = {}
        
        for game in os.listdir(os.path.join(base_dir, model)):
            if game_name in game:
                metric_list = ['return_mean', 'return_std', 'test_return_mean', 'test_return_std']
                # runs = os.listdir(os.path.join(base_dir, model, game))[:-1]
                runs = [run for run in os.listdir(os.path.join(base_dir, model, game)) if run.isdigit()]
                # runs = ['1'] # TODO: delete this, temp solution while exp are running
                n_runs = len(runs)
                for run in runs:
                    data[model][run] = {}

                for metric in metric_list:
                    data[model]['average'][metric] = None

                    for run in runs:
                        metrics, config = get_metrics_config(os.path.join(base_dir, model, game, run))

                        if data[model]['average'][metric] is None:
                            # Get update steps
                            train_update_steps = metrics['return_mean']['steps']
                            test_update_steps = metrics['test_return_mean']['steps']
                            data[model]['average']['train_steps'] = train_update_steps
                            data[model]['average']['test_steps'] = test_update_steps

                            data[model][run]['train_steps'] = train_update_steps
                            data[model][run]['test_steps'] = test_update_steps
                            # Initialize list of average metrics
                            data[model]['average'][metric] = np.array(metrics[metric]['values'])/n_runs
                        else:
                            data[model]['average'][metric] += np.array(metrics[metric]['values'])/n_runs
                        
                        data[model][run][metric] = metrics[metric]['values']
                break
    return data

def create_plots(sacred_directory, models, game_name, test_only, output_dir, output_file_name):
    output_file_name = output_file_name if output_file_name is not None else game_name
    
    # Get data
    data = results_parser(sacred_directory, models, game_name)

    plt.figure(figsize=(10, 10))
    sns.set_theme(style="whitegrid", context="talk")

    if test_only:
        palette = sns.color_palette('Paired', n_colors=len(models)*2)
    else:
        palette = sns.color_palette('tab10', n_colors=len(models))
    colors = cycle(palette)

    for model in models:
        return_mean = data[model]['average']['return_mean']
        return_std = data[model]['average']['return_std']
        test_return_mean = data[model]['average']['test_return_mean']
        test_return_std = data[model]['average']['test_return_std']
        train_steps = data[model]['average']['train_steps']
        test_steps = data[model]['average']['test_steps']
        
        if test_only:
            col = next(colors)
            sns.lineplot(x=train_steps, y=return_mean, label=model+'_train', color=col)
            plt.fill_between(train_steps, return_mean - return_std, return_mean + return_std, color=col, alpha=0.2)
        col = next(colors)
        sns.lineplot(x=test_steps, y=test_return_mean, label=model+'_test', color=col)
        plt.fill_between(test_steps, test_return_mean - test_return_std, test_return_mean + test_return_std, color=col, alpha=0.2)

    plt.xlabel('Training steps')
    plt.ylabel('Average return across runs')
    plt.title(f'Evolution of average return for {game_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    if output_dir is not None:
        plt.savefig(f'{output_dir}/{output_file_name}.png', dpi=400)
        return None
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(f'plots/{output_file_name}.png', dpi=400)


def cli():
    p = argparse.ArgumentParser()
    # Directory conventions
    p.add_argument("--sacred_directory", type=str, default="./")
    p.add_argument("--models", nargs='+', required=True)
    p.add_argument("--game", type=str, required=True)
    p.add_argument("--test_only", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--output_file_name", type=str, default=None)
    args = p.parse_args()

    create_plots(args.sacred_directory, args.models, args.game, args.test_only, args.output_dir, args.output_file_name)

if __name__ == "__main__":
    cli()