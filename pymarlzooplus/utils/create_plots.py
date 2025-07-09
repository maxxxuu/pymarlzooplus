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


def normalize_data(data, models):
    gmin = float('inf')
    gmax = float('-inf')
    for type in ['test_return', 'return']:
        for model in models:
            arr = np.array(data[model]['average'][f'{type}_mean'])
            gmin = min(gmin, np.min(arr))
            gmax = max(gmax, np.max(arr))
        assert gmax > gmin

        for model in models:
            mean = np.array(data[model]['average'][f'{type}_mean'])
            std = np.array(data[model]['average'][f'{type}_std'])
            data[model]['average'][f'min_max_norm_{type}_mean'] = (mean - gmin) / (gmax - gmin)                  
            data[model]['average'][f'min_max_norm_{type}_std'] = (std - gmin) / (gmax - gmin)
    return data

def results_parser(sacred_directory, models, game_name, normalized):
    base_dir = sacred_directory

    metric_list = ['return_mean', 'return_std', 'test_return_mean', 'test_return_std']

    # Create data dictionary for a specific game
    # data[model][run][metric]
    # data[model]['average'][metric]
    data = {}
    for model in models:

        data[model] = {}
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

                        # Initialize list of average metrics that adds values iteratively
                        if data[model]['average'][metric] is None:
                            # Get update steps
                            train_update_steps = metrics['return_mean']['steps']
                            test_update_steps = metrics['test_return_mean']['steps']
                            data[model]['average']['train_steps'] = train_update_steps
                            data[model]['average']['test_steps'] = test_update_steps

                            data[model][run]['train_steps'] = train_update_steps
                            data[model][run]['test_steps'] = test_update_steps

                            # Initialize list of average metrics
                            metric_run_mean = np.array(metrics[metric]['values']) / n_runs
                            data[model]['average'][metric] = metric_run_mean

                            # Normalized episodic reward
                            if metric == 'return_mean' or metric == 'return_std':
                                data[model]['average'][f'normalized_{metric}'] = metric_run_mean / metrics['ep_length_mean']['values']
                            elif metric == 'test_return_mean' or metric == 'test_return_std':
                                data[model]['average'][f'normalized_{metric}'] = metric_run_mean / metrics['test_ep_length_mean']['values']

                        else:
                            metric_run_mean = np.array(metrics[metric]['values']) / n_runs
                            data[model]['average'][metric] += metric_run_mean

                            # Normalized episodic reward
                            if metric == 'return_mean' or metric == 'return_std':
                                data[model]['average'][f'normalized_{metric}'] += metric_run_mean / metrics['ep_length_mean']['values']
                            elif metric == 'test_return_mean' or metric == 'test_return_std':
                                data[model]['average'][f'normalized_{metric}'] += metric_run_mean / metrics['test_ep_length_mean']['values']

                        data[model][run][metric] = metrics[metric]['values']

                break

    if normalized:
        return normalize_data(data, models)
    return data


def palette_choice(models, test_only, no_fill_between):
    # Having Train and Test with fill between is unreadable
    if not test_only and no_fill_between:  # plot Train and Test ; no fill between
        palette = sns.color_palette('Paired', n_colors=len(models) * 2)

    if test_only:
        if not no_fill_between:  # plot Test ; fill between
            palette = sns.color_palette('hls', n_colors=len(models))
        elif no_fill_between:  # plot Test ; no fill between
            palette = sns.color_palette('hls', n_colors=len(models))

    return palette


def create_plots(args):
    # Sacred parameters
    sacred_directory = args.sacred_directory
    models = args.models
    game_name = args.game
    test_only = args.test_only
    # Plot parameters
    linestyle = args.linestyle
    no_fill_between = args.no_fill_between
    normalized = args.normalized
    # Output parameters
    output_dir = args.output_dir
    output_file_name = args.output_file_name if args.output_file_name is not None else game_name

    # Get data
    data = results_parser(sacred_directory, models, game_name, normalized)

    plt.figure(figsize=(10, 10))
    sns.set_theme(style="whitegrid", context="talk")

    models = sorted(models)
    palette = palette_choice(models, test_only, no_fill_between)
    colors = cycle(palette)

    for model in models:
        linestyle = 'dotted' if model == model.split('_')[0] else 'solid'

        return_mean = data[model]['average']['return_mean']
        return_std = data[model]['average']['return_std']
        test_return_mean = data[model]['average']['test_return_mean']
        test_return_std = data[model]['average']['test_return_std']
        train_steps = data[model]['average']['train_steps']
        test_steps = data[model]['average']['test_steps']

        if not test_only:
            col = next(colors)
            sns.lineplot(x=train_steps, y=return_mean, label=model + '_train', color=col, linestyle=linestyle)
            if not no_fill_between:
                plt.fill_between(train_steps, return_mean - return_std, return_mean + return_std, color=col, alpha=0.2)
        col = next(colors)
        sns.lineplot(x=test_steps, y=test_return_mean, label=model + '_test', color=col, linestyle=linestyle)
        if not no_fill_between:
            plt.fill_between(test_steps, test_return_mean - test_return_std, test_return_mean + test_return_std,
                             color=col, alpha=0.2)

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

    if normalized:
        plt.figure(figsize=(10, 10))
        sns.set_theme(style="whitegrid", context="talk")

        models = sorted(models)
        palette = palette_choice(models, test_only, no_fill_between)
        colors = cycle(palette)
        for model in models:
            linestyle = 'dotted' if model == model.split('_')[0] else 'solid'

            return_mean = data[model]['average']['min_max_norm_return_mean']
            return_std = data[model]['average']['min_max_norm_return_std']
            test_return_mean = data[model]['average']['min_max_norm_test_return_mean']
            test_return_std = data[model]['average']['min_max_norm_test_return_std']
            train_steps = data[model]['average']['train_steps']
            test_steps = data[model]['average']['test_steps']

            if not test_only:
                col = next(colors)
                sns.lineplot(x=train_steps, y=return_mean, label=model + '_train', color=col, linestyle=linestyle)
                if not no_fill_between:
                    plt.fill_between(train_steps, return_mean - return_std, return_mean + return_std, color=col, alpha=0.2)
            col = next(colors)
            sns.lineplot(x=test_steps, y=test_return_mean, label=model + '_test', color=col, linestyle=linestyle)
            if not no_fill_between:
                plt.fill_between(test_steps, test_return_mean - test_return_std, test_return_mean + test_return_std,
                                color=col, alpha=0.2)

        plt.xlabel('Training steps')
        plt.ylabel('Average return across runs')
        plt.title(f'Evolution of average return for {game_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()

        if output_dir is not None:
            plt.savefig(f'{output_dir}/norm_{output_file_name}.png', dpi=400)
            return None
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(f'plots/norm_{output_file_name}.png', dpi=400)


def cli():
    p = argparse.ArgumentParser()
    # Directory conventions
    p.add_argument("--sacred_directory", type=str, default="./")
    p.add_argument("--models", nargs='+', required=True)
    p.add_argument("--game", type=str, required=True)
    p.add_argument("--test_only", action="store_true")
    p.add_argument("--linestyle", type=str, default='solid')
    p.add_argument("--no_fill_between", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--output_file_name", type=str, default=None)
    p.add_argument("--normalized", action="store_true")
    args = p.parse_args()

    create_plots(args)


if __name__ == "__main__":
    cli()