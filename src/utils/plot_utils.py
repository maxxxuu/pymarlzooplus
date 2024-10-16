import json
import os
import random
import pickle

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


PREDEFINED_MAP_ALGO_COLORS = {
    'QMIX': '#4169E1',  # Royal blue
    'QPLEX': '#32CD32',  # Lime green
    'MAA2C': '#FF6347',  # Tomato
    'MAPPO': '#40E0D0',  # Turquoise
    'HAPPO': '#AFEEEE',  # Pale turquoise (more vibrant than pastel green)
    'MAT-DEC': '#000000',  # Black
    'COMA': '#9370DB',  # Medium purple (richer than plum)
    'EOI': '#FFD700',  # Gold (bright alternative to yellow)
    'MASER': '#C71585',  # Medium violet red (stronger than light red)
    'EMC': '#A9A9A9',  # Dark gray (more visible than light gray)
    'CDS': '#964B00',  # Brown
}


def create_only_legend(path_to_save):

    # Create a figure and axis for the legend
    fig, ax = plt.subplots(figsize=(12, 1))
    ax.axis('off')  # Turn off axis

    # Create a list of patches to add to the legend
    patches = [
        plt.Line2D([0], [0], color=color, marker='o', markersize=15, label=algo, linestyle='None', markeredgewidth=1.5)
        for algo, color in PREDEFINED_MAP_ALGO_COLORS.items()
    ]

    # Add the legend to the plot
    legend = ax.legend(handles=patches, loc='center', ncol=11, frameon=False, fontsize='large', handletextpad=0.5, columnspacing=1)

    # Save the legend as an image
    plot_path = os.path.join(path_to_save, "MARL_Legend.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)

    # Close the plot
    plt.close()


def base_read_json(json_path):
    """
    json_path: The path to a .json file
    """

    assert os.path.exists(json_path), \
        f"The provided path to json file does not exist! \n'json_path': {json_path}"

    # Open the file for reading
    with open(json_path, 'r') as file:
        # Load data from the file into a Python dictionary
        data = json.load(file)

    return data


def read_json(json_path):
    """
        json_path: The path to info.json file
        """

    assert os.path.basename(json_path) == 'info.json', \
        f"The provided path {json_path} is not a path of a info.json file!"

    try:
        data = base_read_json(json_path)
    except:
        # In case that it fails to load info.json, try metrics.json
        json_base_path = os.path.dirname(json_path)
        json_metrics_path = os.path.join(json_base_path, "metrics.json")
        data = base_read_json(json_metrics_path)
        # Transform data in the form of info.json
        new_data = {}
        for data_key in data.keys():
            data_values = data[data_key]['values']
            data_t = data[data_key]['steps']
            new_data[data_key] = data_values
            new_data[data_key + '_T'] = data_t
        data = new_data

    return data


def create_plot(
        x_data,
        y_data_mean,
        y_data_std,
        path_to_save,
        x_label,
        y_label,
        plot_title,
        legend_labels=None
):

    assert len(x_data) == len(y_data_mean), \
        f"'len(x_data)': {len(x_data)}, 'len(y_data_mean)': {len(y_data_mean)}"
    if y_data_std[0] is not None:
        assert len(y_data_std) == len(y_data_mean), \
            f"'len(y_data_std)': {len(y_data_std)}, 'len(y_data_mean)': {len(y_data_mean)}"

    # Create new figure
    plt.figure()

    for data_idx in range(len(x_data)):

        # Plot the data
        plt.plot(
            x_data[data_idx],
            y_data_mean[data_idx],
            label=None if legend_labels is None else legend_labels[data_idx],
        )

        # Add std if available
        if y_data_std[0] is not None:
            # Calculate the upper and lower bounds of the standard deviation
            std_upper = np.array(y_data_mean[data_idx]) + 1.15*np.array(y_data_std[data_idx])  # 75%
            std_lower = np.array(y_data_mean[data_idx]) - 1.15*np.array(y_data_std[data_idx])  # 75%
            # Add a shaded area for the standard deviation
            plt.fill_between(x_data[data_idx], std_lower, std_upper, alpha=0.2)

    if legend_labels is not None:
        # Adding legend
        plt.legend()

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.tight_layout()

    # Save and close
    plt.savefig(path_to_save)
    plt.close()


def get_mean_and_std_data(results_data, results_type):

    # Mean values
    x_data = results_data[results_type + "_T"]
    mean_data = results_data[results_type]
    # Some metrics are stored in list of dictionaries
    if isinstance(mean_data[0], dict):
        # Extract the 'value' from each dictionary and convert to a numpy array
        if 'value' not in mean_data[0]:  # No values recorded, skip this metric
            return None, None, None
        mean_values = [item['value'] for item in mean_data]
        mean_data = np.array(mean_values, dtype=mean_data[0]['dtype'])

    # Std values
    std_data_key = "_".join(results_type.split("_")[:-1]) + "_std"
    std_data = None if std_data_key not in results_data.keys() else results_data[std_data_key]
    if std_data is not None and isinstance(std_data[0], dict):
        std_values = [item['value'] for item in std_data]
        std_data = np.array(std_values, dtype=std_data[0]['dtype'])

    return x_data, mean_data, std_data


def get_return_data(results_data):

    # Get "return" data for x and y axes
    x_return_data, return_mean_data, return_std_data = \
        get_mean_and_std_data(results_data, "return_mean")
    # Get "test_return" data for x and y axes
    x_test_return_data, test_return_mean_data, test_return_std_data = \
        get_mean_and_std_data(results_data, "test_return_mean")

    # Get "ep_length_mean" to divide "return_mean_data"
    _, ep_length_mean_data, _ = \
        get_mean_and_std_data(results_data, "ep_length_mean")
    # Get "test_ep_length_mean" to divide "test_return_mean_data"
    _, test_ep_length_mean_data, _ = \
        get_mean_and_std_data(results_data, "test_ep_length_mean")

    # Calculate the normalized returns
    assert len(return_mean_data) == len(ep_length_mean_data), \
        f"'len(return_mean_data)': {len(return_mean_data)}, " + \
        f"'len(ep_length_mean_data)': {len(ep_length_mean_data)}"
    normalized_return_mean_data = np.array(return_mean_data) / np.array(ep_length_mean_data)
    normalized_return_std_data = np.array(return_std_data) / np.array(ep_length_mean_data)
    assert len(test_return_mean_data) == len(test_ep_length_mean_data), \
        f"'len(test_return_mean_data)': {len(test_return_mean_data)}, " + \
        f"'len(test_ep_length_mean_data)': {len(test_ep_length_mean_data)}"
    test_normalized_return_mean_data = np.array(test_return_mean_data) / np.array(test_ep_length_mean_data)
    test_normalized_return_std_data = np.array(test_return_std_data) / np.array(test_ep_length_mean_data)

    return (
        x_return_data,
        return_mean_data,
        return_std_data,
        x_test_return_data,
        test_return_mean_data,
        test_return_std_data,
        normalized_return_mean_data,
        normalized_return_std_data,
        test_normalized_return_mean_data,
        test_normalized_return_std_data
    )


def plot_single_experiment_results(path_to_results, algo_name, env_name):
    """
    path_to_results: str, a single path where inside there is the "info.json" file.
    algo_name: str, name of the algorithm, e.g., "qmix"
    env_name: str, name of the environment, e.g., "rware:rware-tiny-2ag-v1"
    """

    assert os.path.exists(path_to_results), \
        f"The provided path to results does not exist! \n'path_to_results': {path_to_results}"

    # Get results
    path_to_info_json = os.path.join(path_to_results, 'info.json')
    results_data = read_json(path_to_info_json)

    plot_title = "Algo: " + algo_name + ", Env: " + env_name
    path_to_save_results = os.path.join(path_to_results, "plots")
    if not os.path.exists(path_to_save_results):
        os.mkdir(path_to_save_results)

    for results_type in results_data.keys():

        # Ignore "timesteps" and "std".
        # These are used only in combination with mean metric values.
        # Also, completely ignore "episode", and "test_return" since it will be plotted with "return".
        if "_T" in results_type or \
           "_std" in results_type or \
           "episode" in results_type or \
           "test_return" in results_type:
            continue

        if "return" in results_type:

            # Get "returns" and normalized "returns" data
            (
                x_return_data,
                return_mean_data,
                return_std_data,
                x_test_return_data,
                test_return_mean_data,
                test_return_std_data,
                normalized_return_mean_data,
                normalized_return_std_data,
                test_normalized_return_mean_data,
                test_normalized_return_std_data
             ) = get_return_data(results_data)

            # Create plot for the unnormalized returns
            path_to_save = os.path.join(path_to_save_results, "return_mean")
            create_plot(
                [x_return_data, x_test_return_data],
                [return_mean_data, test_return_mean_data],
                [return_std_data, test_return_std_data],
                path_to_save,
                "Steps",
                "Episodic Reward",
                plot_title,
                legend_labels=["Train", "Test"]
            )

            # Create plot for the normalized returns
            path_to_save = os.path.join(path_to_save_results, "normalized_return_mean")
            create_plot(
                [x_return_data, x_test_return_data],
                [normalized_return_mean_data, test_normalized_return_mean_data],
                [normalized_return_std_data, test_normalized_return_std_data],
                path_to_save,
                "Steps",
                "Per-Step Episodic Reward",
                plot_title,
                legend_labels=["Train", "Test"]
            )

        else:
            # Get data for x and y axes
            x_data, mean_data, std_data = get_mean_and_std_data(results_data, results_type)
            if mean_data is None:  # No values found, skip it
                continue
            # Define y_label
            results_type_for_y_axis_label = " ".join([elem for elem in results_type.split("_") if elem != "mean"])
            # Define where to save the plot
            path_to_save = os.path.join(path_to_save_results, results_type)
            # Create the plot
            create_plot(
                [x_data],
                [mean_data],
                [std_data],
                path_to_save,
                "Steps",
                results_type_for_y_axis_label,
                plot_title
            )


def calculate_mean_and_std_of_multiple_exps(x_data_list, mean_data_list):

    def truncate_data(x, y, max_time):
        """ Truncate data to minimum max_time """
        valid_indices = x <= max_time
        return x[valid_indices], y[valid_indices]

    # Step 1: Find the maximum common ending timestep
    max_common_time = min([max(x) for x in x_data_list])

    # Step 2: Truncate all datasets
    truncated_data = [
        truncate_data(np.array(x), np.array(y), max_common_time) for x, y in zip(x_data_list, mean_data_list)
    ]

    # Step 3: Define a common set of timesteps and interpolate
    # Increase the data resolution by a factor of 10.
    common_timeline = np.linspace(1, max_common_time, num=len(truncated_data[0][0])*10)
    interpolated_data = np.array([np.interp(common_timeline, x, y) for x, y in truncated_data])

    # Step 4: Calculate mean and standard deviation
    mean_data = np.mean(interpolated_data, axis=0)
    std_data = np.std(interpolated_data, axis=0)

    n_samples = interpolated_data.shape[0]

    return mean_data, std_data, common_timeline, n_samples


def create_multiple_exps_plot(
        all_results,
        path_to_save,
        plot_title,
        legend_labels,  # Algorithm names
        env_name,
        plot_train=True,
        plot_legend_bool=False
):

    # Create new figures, one for returns and another one for per-set returns.
    plt.figure(1)
    plt.xlabel("Steps")
    plt.ylabel("Episodic Reward")
    plt.title(plot_title)

    plt.figure(2)
    plt.xlabel("Steps")
    plt.ylabel("Per-step Episodic Reward")
    plt.title(plot_title)

    lines = []  # To keep track of plot lines for legend
    plot_legends = []  # To keep track of plot legends
    extra_lines = []  # To track lines not in PREDEFINED_MAP_ALGO_COLORS
    extra_plot_legends = []  # To track labels not in PREDEFINED_MAP_ALGO_COLORS

    for alg_idx in range(len(all_results)):

        mean_data = all_results[alg_idx][0]
        std_data = all_results[alg_idx][1]
        norm_mean_data = all_results[alg_idx][2]
        norm_std_data = all_results[alg_idx][3]
        common_timeline = all_results[alg_idx][4]
        test_mean_data = all_results[alg_idx][5]
        test_std_data = all_results[alg_idx][6]
        norm_test_mean_data = all_results[alg_idx][7]
        norm_test_std_data = all_results[alg_idx][8]
        test_common_timeline = all_results[alg_idx][9]
        n_samples = all_results[alg_idx][10]

        # Check data consistency
        assert (len(mean_data) ==
                len(norm_mean_data) ==
                len(common_timeline)), \
            (f"'len(mean_data)': {len(mean_data)}, "
             f"\n'len(norm_mean_data)': {len(norm_mean_data)}, "
             f"\n'len(common_timeline)': {len(common_timeline)}, ")
        assert (len(test_mean_data) ==
                len(norm_test_mean_data) ==
                len(test_common_timeline)), \
            (f"'len(test_mean_data)': {len(test_mean_data)}, "
             f"\n'len(norm_test_mean_data)': {len(norm_test_mean_data)}, "
             f"\n'len(test_common_timeline)': {len(test_common_timeline)}, ")
        if std_data is not None:
            assert (len(std_data) ==
                    len(norm_std_data) ==
                    len(common_timeline)), \
                (f"'len(std_data)': {len(std_data)}, "
                 f"'len(norm_std_data)': {len(norm_std_data)}, "
                 f"'len(common_timeline)': {len(common_timeline)}")
            assert (len(test_std_data) ==
                    len(norm_test_std_data) ==
                    len(test_common_timeline)), \
                (f"'len(test_std_data)': {len(test_std_data)}, "
                 f"'len(norm_test_std_data)': {len(norm_test_std_data)}, "
                 f"'len(test_common_timeline)': {len(test_common_timeline)}")

        data_for_plots = [
            [
                mean_data,
                std_data,
                common_timeline,
                test_mean_data,
                test_std_data,
                test_common_timeline
            ],
            [
                norm_mean_data,
                norm_std_data,
                common_timeline,
                norm_test_mean_data,
                norm_test_std_data,
                test_common_timeline
            ]
        ]

        # Define the label to plot in legend
        plot_legend = legend_labels[alg_idx] if plot_train is False else legend_labels[alg_idx] + "-test"

        # Retrieve color based on 'plot_legend', or generate a random color if label is not in color_map
        color = PREDEFINED_MAP_ALGO_COLORS.get(plot_legend, f'#{random.randint(0, 0xFFFFFF):06x}')

        # Keep the used plot_legend
        if plot_legend in PREDEFINED_MAP_ALGO_COLORS:
            plot_legends.append(plot_legend)
        else:
            extra_plot_legends.append(plot_legend)

        for data_for_plot_idx, data_for_plot in enumerate(data_for_plots):

            # Set which figure to update
            plt.figure(data_for_plot_idx+1)

            # Plot the test data
            line, = plt.plot(
                data_for_plot[5],
                data_for_plot[3],
                label=plot_legend,
                color=color
            )

            # Append to either main or extra lines list
            if plot_legend in PREDEFINED_MAP_ALGO_COLORS:
                if data_for_plot_idx == 0:
                    lines.append([])
                lines[-1].append([line])
            else:
                if data_for_plot_idx == 0:
                    extra_lines.append([])
                extra_lines[-1].append([line])

            # Add std if available
            if data_for_plot[4] is not None:

                # Calculate the upper and lower bounds of the standard deviation
                std_upper = np.array(data_for_plot[3]) + 1.15*np.array(data_for_plot[4]) / np.sqrt(n_samples)  # 75%
                std_lower = np.array(data_for_plot[3]) - 1.15*np.array(data_for_plot[4]) / np.sqrt(n_samples)  # 75%

                # Add a shaded area for the standard deviation
                plt.fill_between(data_for_plot[5], std_lower, std_upper, alpha=0.2, color=color)

            # Plot the train data
            if plot_train is True:
                train_plot_legend = legend_labels[alg_idx] + "-train"
                train_color = f'#{random.randint(0, 0xFFFFFF):06x}'
                line, = plt.plot(
                    data_for_plot[2],
                    data_for_plot[0],
                    label=train_plot_legend,
                    color=train_color
                )

                # Append to either main or extra lines list
                # based on the 'plot_legend' since 'train_plot_legend'
                # is not in 'PREDEFINED_MAP_ALGO_COLORS' by default
                if plot_legend in PREDEFINED_MAP_ALGO_COLORS:
                    lines[-1][-1].append(line)
                else:
                    extra_lines[-1][-1].append(line)

                # Add std if available
                if data_for_plot[1] is not None:
                    # Calculate the upper and lower bounds of the standard deviation
                    std_upper = np.array(data_for_plot[0]) + np.array(data_for_plot[1])
                    std_lower = np.array(data_for_plot[0]) - np.array(data_for_plot[1])
                    # Add a shaded area for the standard deviation
                    plt.fill_between(data_for_plot[2], std_lower, std_upper, alpha=0.2, color=train_color)

    ## Custom legend creation
    legend_order = list(PREDEFINED_MAP_ALGO_COLORS.keys())
    legend_lines_fig_1 = []
    legend_lines_fig_2 = []
    # First should be the lines which are listed in 'legend_order'
    for __plot_legend in legend_order:
        if __plot_legend in plot_legends:
            # Find the index of '__plot_legend' in 'plot_legends'
            _plot_legend_idx = plot_legends.index(__plot_legend)
            # Add the lines for figure 1
            legend_lines_fig_1.extend(lines[_plot_legend_idx][0])
            # Add the lines for figure 2
            legend_lines_fig_2.extend(lines[_plot_legend_idx][1])
    # Then should be all the rest lines (of algorithms not listed in 'legend_order')
    for _lines in extra_lines:
        # Add the lines for figure 1
        legend_lines_fig_1.extend(_lines[0])
        # Add the lines for figure 2
        legend_lines_fig_2.extend(_lines[1])

    # Adding legend, save, and close
    plt.figure(1)
    if plot_legend_bool:
        plt.legend(handles=legend_lines_fig_1)
    plt.tight_layout()
    path_to_save_plot = os.path.join(path_to_save, f"return_mean_env={env_name}")
    plt.savefig(path_to_save_plot)
    plt.close()

    plt.figure(2)
    if plot_legend_bool:
        plt.legend(handles=legend_lines_fig_2)
    plt.tight_layout()
    path_to_save_plot = os.path.join(path_to_save, f"normalized_return_mean_env={env_name}")
    plt.savefig(path_to_save_plot)
    plt.close()


def calculate_mean_best_reward_over_multiple_experiments(
        all_results,
        path_to_save,
        algo_names,
        env_name,
        n_last_values=50
):

    def truncate_data(y, max_time):
        """ Truncate data to minimum max_time """
        y = np.array([y_element[:max_time] for y_element in y])
        return y

    def get_max_indices(_data):
        all_values_best_idx = np.argmax(_data, axis=1)
        last_values_best_idx = np.argmax(np.array(_data)[:, -n_last_values:], axis=1)
        return all_values_best_idx, last_values_best_idx

    def get_best_rewards(_data, all_values_best_idx, last_values_best_idx):

        _best_rewards = {}

        row_indices = np.arange(_data.shape[0])
        _best_rewards['overall_mean_max_reward'] = np.mean(_data[row_indices, all_values_best_idx])
        _best_rewards['overall_std_max_reward'] = np.std(_data[row_indices, all_values_best_idx])
        _best_rewards['overall_median_max_reward'] = np.median(_data[row_indices, all_values_best_idx])
        _best_rewards['overall_25_percentile_max_reward'] = np.percentile(_data[row_indices, all_values_best_idx], 25)
        _best_rewards['overall_75_percentile_max_reward'] = np.percentile(_data[row_indices, all_values_best_idx], 75)
        _best_rewards['overall_min_max_reward'] = np.min(_data[row_indices, all_values_best_idx])
        _best_rewards['overall_max_max_reward'] = np.max(_data[row_indices, all_values_best_idx])

        last_values_data = _data[:, -n_last_values:]
        _best_rewards['last_values_mean_max_reward'] = np.mean(last_values_data[row_indices, last_values_best_idx])
        _best_rewards['last_values_std_max_reward'] = np.std(last_values_data[row_indices, last_values_best_idx])
        _best_rewards['last_values_median_max_reward'] = np.median(last_values_data[row_indices, last_values_best_idx])
        _best_rewards['last_values_25_percentile_max_reward'] = \
            np.percentile(last_values_data[row_indices, last_values_best_idx], 25)
        _best_rewards['last_values_75_percentile_max_reward'] = \
            np.percentile(last_values_data[row_indices, last_values_best_idx], 75)
        _best_rewards['last_values_min_max_reward'] = np.min(last_values_data[row_indices, last_values_best_idx])
        _best_rewards['last_values_max_max_reward'] = np.max(last_values_data[row_indices, last_values_best_idx])

        # Round the results
        for key, value in _best_rewards.items():
            _best_rewards[key] = round(value, 2)

        return _best_rewards

    # Prepare the csv columns and data. These metrics have the same order as the 'best_rewards'
    metrics = [
        'test_overall_mean_max_reward',
        'test_overall_std_max_reward',
        'test_overall_median_max_reward',
        'test_overall_25_percentile_max_reward',
        'test_overall_75_percentile_max_reward',
        'test_overall_min_max_reward',
        'test_overall_max_max_reward',
        'test_last_values_mean_max_reward',
        'test_last_values_std_max_reward',
        'test_last_values_median_max_reward',
        'test_last_values_25_percentile_max_reward',
        'test_last_values_75_percentile_max_reward',
        'test_last_values_min_max_reward',
        'test_last_values_max_max_reward',
        'train_overall_mean_max_reward',
        'train_overall_std_max_reward',
        'train_overall_median_max_reward',
        'train_overall_25_percentile_max_reward',
        'train_overall_75_percentile_max_reward',
        'train_overall_min_max_reward',
        'train_overall_max_max_reward',
        'train_last_values_mean_max_reward',
        'train_last_values_std_max_reward',
        'train_last_values_median_max_reward',
        'train_last_values_25_percentile_max_reward',
        'train_last_values_75_percentile_max_reward',
        'train_last_values_min_max_reward',
        'train_last_values_max_max_reward',
        'test_norm_overall_mean_max_reward',
        'test_norm_overall_std_max_reward',
        'test_norm_overall_median_max_reward',
        'test_norm_overall_25_percentile_max_reward',
        'test_norm_overall_75_percentile_max_reward',
        'test_norm_overall_min_max_reward',
        'test_norm_overall_max_max_reward',
        'test_norm_last_values_mean_max_reward',
        'test_norm_last_values_std_max_reward',
        'test_norm_last_values_median_max_reward',
        'test_norm_last_values_25_percentile_max_reward',
        'test_norm_last_values_75_percentile_max_reward',
        'test_norm_last_values_min_max_reward',
        'test_norm_last_values_max_max_reward',
        'train_norm_overall_mean_max_reward',
        'train_norm_overall_std_max_reward',
        'train_norm_overall_median_max_reward',
        'train_norm_overall_25_percentile_max_reward',
        'train_norm_overall_75_percentile_max_reward',
        'train_norm_overall_min_max_reward',
        'train_norm_overall_max_max_reward',
        'train_norm_last_values_mean_max_reward',
        'train_norm_last_values_std_max_reward',
        'train_norm_last_values_median_max_reward',
        'train_norm_last_values_25_percentile_max_reward',
        'train_norm_last_values_75_percentile_max_reward',
        'train_norm_last_values_min_max_reward',
        'train_norm_last_values_max_max_reward',
    ]
    # Create an empty DataFrame with metrics as the index
    df = pd.DataFrame(index=metrics, columns=algo_names)

    for alg_idx in range(len(all_results)):

        mean_data = all_results[alg_idx][11]
        norm_mean_data = all_results[alg_idx][12]
        test_mean_data = all_results[alg_idx][13]
        norm_test_mean_data = all_results[alg_idx][14]

        # Step 1: Find the maximum common ending timestep
        max_common_time = min([len(x) for x in mean_data])
        test_max_common_time = min([len(x) for x in test_mean_data])

        # Step 2: Truncate all datasets
        data_for_reward_calculation = [
            truncate_data(test_mean_data, test_max_common_time),
            truncate_data(mean_data, max_common_time),
            truncate_data(norm_test_mean_data, test_max_common_time),
            truncate_data(norm_mean_data, max_common_time)
        ]

        data_columns = [
            'test',
            'train',
            'norm test',
            'norm train'
        ]

        test_all_values_best_idx = None
        test_last_values_best_idx = None
        train_all_values_best_idx = None
        train_last_values_best_idx = None
        algo_data_for_csv = np.zeros((len(metrics),), dtype=np.float32)
        for data_idx, (data, data_column) in enumerate(zip(data_for_reward_calculation, data_columns)):

            # Get indices from all experiments of the best reward over the last "n_last_values" and over all values
            if data_column == 'test':
                test_all_values_best_idx, test_last_values_best_idx = get_max_indices(data)
            elif data_column == 'train':
                train_all_values_best_idx, train_last_values_best_idx = get_max_indices(data)

            # Get the best reward statistics
            best_rewards = None
            if 'test' in data_column:
                best_rewards = get_best_rewards(data, test_all_values_best_idx, test_last_values_best_idx)
            elif 'train' in data_column:
                best_rewards = get_best_rewards(data, train_all_values_best_idx, train_last_values_best_idx)
            else:
                raise ValueError(f'data_column: {data_column}')

            # Assign the statistics to the csv data
            algo_data_for_csv[(data_idx * len(best_rewards)): ((data_idx + 1) * len(best_rewards))] = \
                list(best_rewards.values())

        # Assign the algo values to the dataframe
        df[algo_names[alg_idx]] = algo_data_for_csv.copy()

    # Save dataframe
    file_path = os.path.join(path_to_save, f'best_rewards_env={env_name}.csv')
    df.to_csv(file_path)


def plot_multiple_experiment_results(
        paths_to_results,
        algo_names,
        env_name,
        path_to_save,
        plot_train,
        plot_legend_bool
):
    """
    path_to_results: list of str, all paths of the algorithms results for a specific environment.
                     Each path should contain folders like: 1, 2, 3, e.t.c., where each one should have inside
                     a file "info.json". NOTE: The order of paths should be aligned with the other of "algo_names" list.
    algo_names: list of str, all the algorithm names, e.g., ["qmix", "qplex", "maa2c", ...]
    env_name: str, name of the environment, e.g., "rware:rware-tiny-2ag-v1" .
    path_to_save: str, path to save the plots.
    plot_train: bool, whether to plot the training returns or not.
    """

    assert len(paths_to_results) == len(algo_names), \
        f"'len(paths_to_results)': {len(paths_to_results)}, \n'len(algo_names)': {len(algo_names)}"

    all_results = []
    for path_to_results_idx, path_to_results in enumerate(paths_to_results):

        # Check if the provided path is valid
        assert os.path.exists(path_to_results), \
            f"The provided 'path_to_results' does not exist! 'path_to_results': {path_to_results}"

        path_to_exps = [
            os.path.join(path_to_results, elem) for elem in os.listdir(path_to_results) if elem.isdigit()
        ]
        x_return_data_list = []
        return_mean_data_list = []
        x_test_return_data_list = []
        test_return_mean_data_list = []
        normalized_return_mean_data_list = []
        test_normalized_return_mean_data_list = []
        for path_to_exp in path_to_exps:

            # Get results
            path_to_info_json = os.path.join(path_to_exp, 'info.json')
            results_data = read_json(path_to_info_json)

            # Get "returns" and normalized "returns" data
            (
                x_return_data,
                return_mean_data,
                _,
                x_test_return_data,
                test_return_mean_data,
                _,
                normalized_return_mean_data,
                _,
                test_normalized_return_mean_data,
                _
            ) = get_return_data(results_data)

            # Keep returns to compute their mean and std
            x_return_data_list.append(x_return_data)
            return_mean_data_list.append(return_mean_data)
            x_test_return_data_list.append(x_test_return_data)
            test_return_mean_data_list.append(test_return_mean_data)
            normalized_return_mean_data_list.append(normalized_return_mean_data)
            test_normalized_return_mean_data_list.append(test_normalized_return_mean_data)

        (
            mean_data,
            std_data,
            common_timeline,
            n_samples
        ) = calculate_mean_and_std_of_multiple_exps(x_return_data_list, return_mean_data_list)
        (
            norm_mean_data,
            norm_std_data,
            _,
            _
        ) = calculate_mean_and_std_of_multiple_exps(x_return_data_list, normalized_return_mean_data_list)
        (
            test_mean_data,
            test_std_data,
            test_common_timeline,
            _
        ) = calculate_mean_and_std_of_multiple_exps(x_test_return_data_list, test_return_mean_data_list)
        (
            norm_test_mean_data,
            norm_test_std_data,
            _,
            _
        ) = calculate_mean_and_std_of_multiple_exps(x_test_return_data_list, test_normalized_return_mean_data_list)
        all_results.append([
            mean_data, std_data, norm_mean_data, norm_std_data, common_timeline,
            test_mean_data, test_std_data, norm_test_mean_data, norm_test_std_data, test_common_timeline,
            n_samples,
            return_mean_data_list, normalized_return_mean_data_list,
            test_return_mean_data_list, test_normalized_return_mean_data_list
        ])

    # Create plots
    plot_title = env_name
    if os.path.exists(path_to_save) is False:
        os.makedirs(path_to_save)
    create_multiple_exps_plot(
        all_results,
        path_to_save,
        plot_title,
        algo_names,
        env_name,
        plot_train=plot_train,
        plot_legend_bool=plot_legend_bool
    )

    # Create csv file with the mean best rewards
    calculate_mean_best_reward_over_multiple_experiments(
        all_results,
        path_to_save,
        algo_names,
        env_name
    )

    ## Save 'all_results' to use them for extracting average plots per algorithm aver all tasks of each benchmark
    pickle_file_path = os.path.join(path_to_save, f"all_results_env={env_name}.pkl")
    # Open a pickle file for writing
    with open(pickle_file_path, 'wb') as file:
        # Serialize the dictionary and write it to the file
        pickle.dump({"all_results_list": all_results, "algo_names": algo_names, "env_name": env_name}, file)

    print("\nMultiple-experiment plots created successfully! "
          f"\nSaved at: {path_to_save}")


def plot_average_per_algo_for_all_tasks_of_a_benchmark(
        paths_to_pickle_results,
        plot_title,
        path_to_save,
        plot_legend_bool
):
    """
    paths_to_pickle_results: are pickle files, each of which has 3 elements:
        - all_results_list
        - algo_names
        - env_name
    The 'all_results_list' element contains the results of each algo
    in the same order as in their names in 'algo_names'.
    For each algo, the following elements are stored in a list:
        0) mean_data
        1) std_data
        2) norm_mean_data
        3) norm_std_data
        4) common_timeline
        5) test_mean_data
        6) test_std_data
        7) norm_test_mean_data
        8) norm_test_std_data
        9) test_common_timeline
        10) n_samples
        11) return_mean_data_list
        12) normalized_return_mean_data_list
        13) test_return_mean_data_list
        14) test_normalized_return_mean_data_list
    """

    all_results_dict = {}
    all_common_timelines_dict = {}
    for path_to_pickle_task_results in paths_to_pickle_results:
        # Read the pickle file
        with open(path_to_pickle_task_results, 'rb') as file:
            pickle_data = pickle.load(file)
            all_algo_results = pickle_data['all_results_list']
            algo_names = pickle_data['algo_names']

        temp_max = -np.inf
        temp_min = np.inf
        for algo_name_idx, algo_name in enumerate(algo_names):
            algo_mean_results = all_algo_results[algo_name_idx][5]  # test mean data
            # Keep min and max to normalize the values across different tasks
            algo_max = np.max(algo_mean_results)
            algo_min = np.min(algo_mean_results)
            if temp_max < algo_max:
                temp_max = algo_max
            if temp_min > algo_min:
                temp_min = algo_min

        assert temp_min != np.inf, f'temp_min: {temp_min}'
        assert temp_max != -np.inf

        ## Re-iterate over algo results and normalize the values
        for algo_name_idx, algo_name in enumerate(algo_names):
            if algo_name not in list(all_results_dict.keys()):
                all_results_dict[algo_name] = []
            if algo_name not in list(all_common_timelines_dict.keys()):
                all_common_timelines_dict[algo_name] = []
            algo_mean_results = all_algo_results[algo_name_idx][5]  # test mean data
            # Normalize
            norm_algo_mean_results = (algo_mean_results - temp_min) / (temp_max - temp_min)
            # Store to dict
            all_results_dict[algo_name].append(norm_algo_mean_results.copy())
            # Store common_timeline to dict
            all_common_timelines_dict[algo_name].append(all_algo_results[0][9])  # test common timeline

    # Calculate the minimum common timeline for each algo
    for alg_name, alg_common_timelines in all_common_timelines_dict.items():
        temp_min_common_timeline = alg_common_timelines[0]
        for alg_common_timeline in alg_common_timelines:
            if len(temp_min_common_timeline) > len(alg_common_timeline):
                temp_min_common_timeline = alg_common_timeline
        all_common_timelines_dict[alg_name] = temp_min_common_timeline

    # Create new figures, one for returns and another one for per-set returns.
    plt.figure()
    plt.xlabel("Steps")
    plt.ylabel("Normalized Episodic Reward")
    plt.title(plot_title)

    lines = []  # To keep track of plot lines for legend
    plot_legends = []  # To keep track of plot legends
    extra_lines = []  # To track lines not in PREDEFINED_MAP_ALGO_COLORS
    extra_plot_legends = []  # To track labels not in PREDEFINED_MAP_ALGO_COLORS

    for alg_name, alg_results in all_results_dict.items():

        # Define the label to plot in legend
        plot_legend = alg_name

        # Retrieve color based on 'plot_legend', or generate a random color if label is not in color_map
        color = PREDEFINED_MAP_ALGO_COLORS.get(plot_legend, f'#{random.randint(0, 0xFFFFFF):06x}')

        # Keep the used plot_legend
        if plot_legend in PREDEFINED_MAP_ALGO_COLORS:
            plot_legends.append(plot_legend)
        else:
            extra_plot_legends.append(plot_legend)

        # Keep the results according to the minimum common timeline
        min_common_time = min(min([len(x) for x in alg_results]), len(all_common_timelines_dict[alg_name]))
        alg_results = [alg_result[:min_common_time] for alg_result in alg_results]

        # Calculate mean
        mean_alg_results = np.mean(alg_results, axis=0)

        # Plot the test data
        line, = plt.plot(
            all_common_timelines_dict[alg_name][:min_common_time],
            mean_alg_results,
            label=plot_legend,
            color=color
        )

        # Append to either main or extra lines list
        if plot_legend in PREDEFINED_MAP_ALGO_COLORS:
            lines.append(line)
        else:
            extra_lines.append(line)

        ## Add std
        # Calculate standard deviation
        std_alg_results = np.std(alg_results, axis=0)
        # Calculate the upper and lower bounds of the standard deviation
        n_samples = len(alg_results)
        std_upper = mean_alg_results + 1.15*std_alg_results / np.sqrt(n_samples)  # 75%
        std_lower = mean_alg_results - 1.15*std_alg_results / np.sqrt(n_samples)  # 75%
        # Add a shaded area for the standard deviation
        plt.fill_between(all_common_timelines_dict[alg_name][:min_common_time], std_lower, std_upper, alpha=0.2, color=color)

    ## Custom legend creation
    legend_order = list(PREDEFINED_MAP_ALGO_COLORS.keys())
    legend_lines_fig = []
    # First should be the lines which are listed in 'legend_order'
    for _plot_legend in legend_order:
        if _plot_legend in plot_legends:
            # Find the index of '_plot_legend' in 'plot_legends'
            _plot_legend_idx = plot_legends.index(_plot_legend)
            # Add the lines for figure
            legend_lines_fig.extend([lines[_plot_legend_idx]])
    # Then should be all the rest lines (of algorithms not listed in 'legend_order')
    for _line in extra_lines:
        # Add the lines for figure
        legend_lines_fig.extend([_line])

    # Adding legend
    if plot_legend_bool:
        plt.legend(handles=legend_lines_fig)
    plt.tight_layout()

    # Save and close
    if os.path.exists(path_to_save) is False:
        os.makedirs(path_to_save)
    path_to_save_plot = os.path.join(path_to_save, f"benchmark={plot_title}")
    plt.savefig(path_to_save_plot)
    plt.close()


if __name__ == '__main__':

    ## Single algo

    # path_to_results_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/pistonball_v6/emc_pistonball_v6_w_resnet18_wo_rew_stand_results/sacred/emc/pistonball_v6/1"
    # algo_name_ = "emc"
    # env_name_ = "pistonball_v6"
    # plot_single_experiment_results(path_to_results_, algo_name_, env_name_)

    ## Many algos

    # Pistonball
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/pistonball_v6/coma_pistonball_v6_w_parallel_2_threads_w_resnet18_results/sacred/coma/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/pistonball_v6/pistonball_v6_w_parallel_2_threads_w_resnet18_results/sacred/maa2c/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/pistonball_v6/mappo_pistonball_v6_w_parallel_2_threads_w_resnet18_results/results/sacred/mappo/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/pistonball_v6/pistonball_v6_w_resnet18_results/results/sacred/qmix/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/pistonball_v6/pistonball_v6_w_episode_w_resnet18_results/sacred/eoi/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/pistonball_v6/pistonball_v6_w_resnet18_results/results/sacred/qplex/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/pistonball_v6/maser_pistonball_v6_w_resnet18_results/sacred/maser/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/pistonball_v6/cds_pistonball_v6_w_resnet18_results/sacred/cds/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/pistonball_v6/mat_dec_pistonball_v6_w_parallel_2_threads_w_resnet18_results/sacred/mat_dec/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/pistonball_v6/emc_pistonball_v6_w_resnet18_w_rew_stand_results/sacred/emc/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/pistonball_v6/happo_pistonball_v6_w_parallel_2_threads_w_resnet18_results/sacred/happo/pistonball_v6"
    # ]
    # algo_names_ = ["COMA", "MAA2C", "MAPPO", "QMIX", "EOI", "QPLEX", "MASER", "CDS", "MAT-DEC", "EMC", "HAPPO"]
    # env_name_ = "pistonball_v6"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/pistonball_v6/"

    # Pistonball - Ablation
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/pistonball_v6/pistonball_v6_w_parallel_2_threads_w_resnet18_results/sacred/maa2c/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/pistonball_v6/maa2c_pistonball_v6_w_parallel_2_threads_w_trainable_cnn_results/sacred/maa2c/pistonball_v6",
    # ]
    #
    # algo_names_ = ["MAA2C-ResNet18", "MAA2C-CNN"]
    # env_name_ = "Pistonball"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/pistonball_v6_ablation/"

    # Cooperative pong
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/cooperative_pong_v5/coma_cooperative_pong_v5_w_parallel_2_threads_w_resnet18_results/sacred/coma/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/cooperative_pong_v5/cooperative_pong_v5_w_parallel_2_threads_w_resnet18_results/sacred/maa2c/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/cooperative_pong_v5/mappo_cooperative_pong_v5_w_parallel_2_threads_w_resnet18_results/sacred/mappo/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/cooperative_pong_v5/cooperative_pong_v5_w_resnet18_results/sacred/qmix/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/cooperative_pong_v5/eoi_cooperative_pong_v5_w_episode_w_resnet18_results/sacred/eoi/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/cooperative_pong_v5/cooperative_pong_v5_w_resnet18_results/sacred/qplex/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/cooperative_pong_v5/maser_cooperative_pong_v5_w_resnet18_results/sacred/maser/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/cooperative_pong_v5/cooperative_pong_v5_w_resnet18_wo_rew_stand_results/sacred/emc/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/cooperative_pong_v5/cooperative_pong_v5_w_parallel_2_threads_w_resnet18_results/sacred/happo/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/cooperative_pong_v5/cds_cooperative_pong_v5_w_resnet18_results/sacred/cds/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/cooperative_pong_v5/mat_dec_cooperative_pong_v5_w_parallel_2_threads_w_resnet18_results/sacred/mat_dec/cooperative_pong_v5"
    #                    ]
    # algo_names_ = ["COMA", "MAA2C", "MAPPO", "QMIX", "EOI", "QPLEX", "MASER", "EMC", "HAPPO", "CDS", "MAT-DEC"]
    # env_name_ = "cooperative_pong_v5"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/cooperative_pong_v5/"

    # Cooperative pong - Ablation
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/cooperative_pong_v5/cooperative_pong_v5_w_parallel_2_threads_w_resnet18_results/sacred/maa2c/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/cooperative_pong_v5/cooperative_pong_v5_w_parallel_2_threads_w_trainable_cnn_results/sacred/maa2c/cooperative_pong_v5"
    #                    ]
    # algo_names_ = ["MAA2C-ResNet18", "MAA2C-CNN"]
    # env_name_ = "cooperative_pong_v5"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/cooperative_pong_v5_ablation/"

    # Entombed cooperative
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/entombed_cooperative_v3/emc_fixed_entombed_cooperative_v3_w_max_cycles=3500_w_resnet18_results/sacred/emc/entombed_cooperative_v3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/entombed_cooperative_v3/coma_entombed_cooperative_v3_w_parallel_2_threads_w_resnet18_results/results/sacred/coma/entombed_cooperative_v3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/entombed_cooperative_v3/maa2c_entombed_cooperative_v3_w_parallel_2_threads_w_resnet18_results/results/sacred/maa2c/entombed_cooperative_v3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/entombed_cooperative_v3/entombed_cooperative_v3_parallel_2_threads_w_resnet_18_results/results/sacred/mappo/entombed_cooperative_v3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/entombed_cooperative_v3/eoi_entombed_cooperative_v3_w_episode_w_resnet18_results/results/sacred/eoi/entombed_cooperative_v3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/entombed_cooperative_v3/qmix_entombed_cooperative_v3_w_max_cycles=2500_w_resnet18_buffer_size=2000_results/sacred/qmix/entombed_cooperative_v3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/entombed_cooperative_v3/entombed_cooperative_v3_w_max_cycles=2500_w_resnet18_buffer_size=1500_results/sacred/qplex/entombed_cooperative_v3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/entombed_cooperative_v3/maser_fixed_entombed_cooperative_v3_w_max_cycles=3500_w_resnet18_results/sacred/maser/entombed_cooperative_v3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/entombed_cooperative_v3/cds_fixed_entombed_cooperative_v3_w_max_cycles=3500_w_resnet18_results/sacred/cds/entombed_cooperative_v3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/entombed_cooperative_v3/happo_fixed_entombed_cooperative_v3_w_parallel_10_threads_w_resnet18_results/sacred/happo/entombed_cooperative_v3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/entombed_cooperative_v3/mat_dec_fixed_entombed_cooperative_v3_w_parallel_10_threads_w_resnet18_results/sacred/mat_dec/entombed_cooperative_v3"
    #                     ]
    # algo_names_ = ["EMC", "COMA", "MAA2C", "MAPPO", "EOI", "QMIX", "QPLEX", "MASER", "CDS", "HAPPO", "MAT-DEC"]
    # env_name_ = "entombed_cooperative_v3"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/entombed_cooperative_v3/"

    # Space invaders
    # paths_to_results_ = ["/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/space_invaders_v2/space_invaders_v2_parallel_2_threads_w_resnet_18_results/sacred/maa2c/space_invaders_v2",
    #                      "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/space_invaders_v2/space_invaders_v2_parallel_2_threads_w_resnet_18_results/results/sacred/mappo/space_invaders_v2",
    #                      "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/space_invaders_v2/space_invaders_v2_parallel_2_threads_w_resnet_18_results/sacred/coma/space_invaders_v2",
    #                      "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/space_invaders_v2/qmix_space_invaders_v2_w_max_cycles=2500_w_resnet18_buffer_size=2000_results/sacred/qmix/space_invaders_v2",
    #                      "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/space_invaders_v2/eoi_space_invaders_v2_w_episode_w_resnet18_results/results/sacred/eoi/space_invaders_v2",
    #                      "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/space_invaders_v2/space_invaders_v2_w_resnet18_results/sacred/qplex/space_invaders_v2",
    #                      ]
    # algo_names_ = ["MAA2C", "MAPPO", "COMA", "QMIX", "EOI", "QPLEX"]
    # env_name_ = "space_invaders_v2"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/space_invaders_v2/"

    # Cramped Room
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/cramped_room/emc_cramped_room_results/sacred/emc/cramped_room",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/cramped_room/cramped_room_w_parallel_2_threads_results/results/sacred/maa2c/cramped_room",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/cramped_room/mappo_cramped_room_w_parallel_2_threads_results/sacred/mappo/cramped_room",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/cramped_room/coma_cramped_room_w_parallel_2_threads_results/sacred/coma/cramped_room",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/cramped_room/qmix_cramped_room_results/sacred/qmix/cramped_room",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/cramped_room/cramped_room_results/sacred/qplex/cramped_room",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/cramped_room/cramped_room_w_episode_results/sacred/eoi/cramped_room",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/cramped_room/maser_cramped_room_results/sacred/maser/cramped_room",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/cramped_room/cds_cramped_room_results/sacred/cds/cramped_room",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/cramped_room/happo_cramped_room_w_parallel_2_threads_results/sacred/happo/cramped_room",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/cramped_room/mat_dec_cramped_room_w_parallel_2_threads_results/sacred/mat_dec/cramped_room",
    # ]
    # algo_names_ = ["EMC", "MAA2C", "MAPPO", "COMA", "QMIX", "QPLEX", "EOI", "MASER", "CDS", "HAPPO", "MAT-DEC"]
    # env_name_ = "Cramped Room"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/cramped_room/"

    # Assymetric_advantages
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/assymetric_advantages/maser_assymetric_advantages_w_max_steps=100M_results/sacred/maser/asymmetric_advantages",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/assymetric_advantages/maa2c_assymetric_advantages_w_max_steps=100M_w_parallel_2_threads_results/results/sacred/maa2c/asymmetric_advantages",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/assymetric_advantages/mappo_assymetric_advantages_w_max_steps=100M_w_parallel_2_threads_results/sacred/mappo/asymmetric_advantages",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/assymetric_advantages/coma_assymetric_advantages_w_max_steps=100M_w_parallel_2_threads_results/sacred/coma/asymmetric_advantages",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/assymetric_advantages/qmix_assymetric_advantages_w_max_steps=100M_results/sacred/qmix/asymmetric_advantages",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/assymetric_advantages/qplex_assymetric_advantages_w_max_steps=100M_results/sacred/qplex/asymmetric_advantages",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/assymetric_advantages/eoi_assymetric_advantages_w_max_steps=100M_w_episode_results/sacred/eoi/asymmetric_advantages",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/assymetric_advantages/cds_assymetric_advantages_w_max_steps=100M_results/sacred/cds/asymmetric_advantages",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/assymetric_advantages/mat_dec_assymetric_advantages_w_max_steps=100M_w_parallel_2_threads_results/sacred/mat_dec/asymmetric_advantages",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/assymetric_advantages/happo_assymetric_advantages_w_max_steps=100M_w_parallel_2_threads_results/sacred/happo/asymmetric_advantages",
    # ]
    # algo_names_ = ["MASER", "MAA2C", "MAPPO", "COMA", "QMIX", "QPLEX", "EOI", "CDS", "MAT-DEC", "HAPPO"]
    # env_name_ = "Asymmetric Advantages"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/assymetric_advantages/"

    # Coordination ring
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/coordination_ring/maa2c_coordination_ring_w_max_steps=100M_w_parallel_2_threads_results/sacred/maa2c/coordination_ring",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/coordination_ring/mappo_coordination_ring_w_max_steps=100M_w_parallel_2_threads_results/sacred/mappo/coordination_ring",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/coordination_ring/coma_coordination_ring_w_max_steps=100M_w_parallel_2_threads_results/sacred/coma/coordination_ring",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/coordination_ring/coordination_ring_w_max_steps=100M_results/sacred/qmix/coordination_ring",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/coordination_ring/coordination_ring_w_max_steps=100M_results/sacred/qplex/coordination_ring",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/coordination_ring/eoi_coordination_ring_w_max_steps=100M_w_episode_results/sacred/eoi/coordination_ring",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/coordination_ring/maser_coordination_ring_w_max_steps=100M_results/sacred/maser/coordination_ring",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/coordination_ring/happo_coordination_ring_w_parallel_2_threads_results/sacred/happo/coordination_ring",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/coordination_ring/cds_coordination_ring_results/sacred/cds/coordination_ring",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/coordination_ring/mat_dec_coordination_ring_w_max_steps=100M_w_parallel_2_threads_results/sacred/mat_dec/coordination_ring"
    # ]
    # algo_names_ = ["MAA2C", "MAPPO", "COMA", "QMIX", "QPLEX", "EOI", "MASER", "HAPPO", "CDS", "MAT-DEC"]
    # env_name_ = "Coordination Ring"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/coordination_ring/"

    # PressurePlate - linear-4p-v0
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/pressureplate_linear-4p-v0/emc_linear-4p-v0_results/sacred/emc/pressureplate-linear-4p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/pressureplate_linear-4p-v0/qmix_linear-4p-v0_results/sacred/qmix/pressureplate-linear-4p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/pressureplate_linear-4p-v0/maa2c_linear-4p-v0_w_parallel_2_threads_results/sacred/maa2c/pressureplate-linear-4p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/pressureplate_linear-4p-v0/coma_linear-4p-v0_results/sacred/coma/pressureplate-linear-4p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/pressureplate_linear-4p-v0/qplex_linear-4p-v0_results/sacred/qplex/pressureplate-linear-4p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/pressureplate_linear-4p-v0/mappo_linear-4p-v0_w_parallel_2_threads_results/sacred/mappo/pressureplate-linear-4p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/pressureplate_linear-4p-v0/eoi_linear-4p-v0_w_parallel_2_threads_results/sacred/eoi/pressureplate-linear-4p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/pressureplate_linear-4p-v0/maser_linear-4p-v0_results/sacred/maser/pressureplate-linear-4p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/pressureplate_linear-4p-v0/cds_linear-4p-v0_results/sacred/cds/pressureplate-linear-4p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/pressureplate_linear-4p-v0/mat_dec_linear-4p-v0_w_parallel_10_threads_results/sacred/mat_dec/pressureplate-linear-4p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/pressureplate_linear-4p-v0/happo_linear-4p-v0_w_parallel_10_threads_results/sacred/happo/pressureplate-linear-4p-v0"
    # ]
    # algo_names_ = ["EMC", "QMIX", "MAA2C", "COMA", "QPLEX", "MAPPO", "EOI", "MASER", "CDS", "MAT-DEC", "HAPPO"]
    # env_name_ = "pressureplate-linear-4p-v0"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/pressureplate-linear-4p-v0/"

    # PressurePlate - linear-6p-v0
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/pressureplate_linear-6p-v0/emc_linear-6p-v0_results/sacred/emc/pressureplate-linear-6p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/pressureplate_linear-6p-v0/maa2c_linear-6p-v0_w_parallel_2_threads_results/sacred/maa2c/pressureplate-linear-6p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/pressureplate_linear-6p-v0/qmix_linear-6p-v0_results/sacred/qmix/pressureplate-linear-6p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/pressureplate_linear-6p-v0/coma_linear-6p-v0_results/sacred/coma/pressureplate-linear-6p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/pressureplate_linear-6p-v0/qplex_linear-6p-v0_results/sacred/qplex/pressureplate-linear-6p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/pressureplate_linear-6p-v0/eoi_linear-6p-v0_w_parallel_2_threads_results/sacred/eoi/pressureplate-linear-6p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/pressureplate_linear-6p-v0/mappo_linear-6p-v0_w_parallel_2_threads_results/sacred/mappo/pressureplate-linear-6p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/pressureplate_linear-6p-v0/maser_linear-6p-v0_results/sacred/maser/pressureplate-linear-6p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/pressureplate_linear-6p-v0/cds_linear-6p-v0_results/sacred/cds/pressureplate-linear-6p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/pressureplate_linear-6p-v0/happo_linear-6p-v0_w_parallel_2_threads_results/sacred/happo/pressureplate-linear-6p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/pressureplate_linear-6p-v0/mat_dec_linear-6p-v0_w_parallel_2_threads_results/sacred/mat_dec/pressureplate-linear-6p-v0",    ]
    # algo_names_ = ["EMC", "MAA2C", "QMIX", "COMA", "QPLEX", "EOI", "MAPPO", "MASER", "CDS", "HAPPO", "MAT-DEC"]
    # env_name_ = "pressureplate-linear-6p-v0"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/pressureplate-linear-6p-v0/"

    # LBF - 2s-11x11-3p-2f
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/lbf_2s-11x11-3p-2f/cds_lbf_2s-11x11-3p-2f_results/sacred/cds/lbforaging:Foraging-2s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/lbf_2s-11x11-3p-2f/maser_lbf_2s-11x11-3p-2f_results/sacred/maser/lbforaging:Foraging-2s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/lbf_2s-11x11-3p-2f/qplex_lbf_2s-11x11-3p-2f_results/sacred/qplex/lbforaging:Foraging-2s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/lbf_2s-11x11-3p-2f/qmix_lbf_2s-11x11-3p-2f_results/sacred/qmix/lbforaging:Foraging-2s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/lbf_2s-11x11-3p-2f/happo_lbf_2s-11x11-3p-2f_w_parallel_10_threads_results/sacred/happo/lbforaging:Foraging-2s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/lbf_2s-11x11-3p-2f/coma_lbf_2s-11x11-3p-2f_w_parallel_10_threads_results/Foraging-2s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/lbf_2s-11x11-3p-2f/eoi_lbf_2s-11x11-3p-2f_w_episode_results/Foraging-2s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/lbf_2s-11x11-3p-2f/mat_dec_lbf_2s-11x11-3p-2f_w_parallel_10_threads_results/sacred/mat_dec/lbforaging:Foraging-2s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/lbf_2s-11x11-3p-2f/mappo_lbf_2s-11x11-3p-2f_w_parallel_10_threads_results/Foraging-2s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/lbf_2s-11x11-3p-2f/maa2c_lbf_2s-11x11-3p-2f_w_parallel_10_threads_results/Foraging-2s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/lbf_2s-11x11-3p-2f/emc_lbf_2s-11x11-3p-2f_w_rew_stand_results/sacred/emc/lbforaging:Foraging-2s-11x11-3p-2f-coop-v2"
    # ]
    # algo_names_ = ["CDS", "MASER", "QPLEX", "QMIX", "HAPPO", "COMA", "EOI", "MAT-DEC", "MAPPO", "MAA2C", "EMC"]
    # env_name_ = "lbf_2s-11x11-3p-2f"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/lbf_2s-11x11-3p-2f/"

    # LBF - 4s-11x11-3p-2f
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/lbf_4s-11x11-3p-2f/cds_lbf_4s-11x11-3p-2f_results/sacred/cds/lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/lbf_4s-11x11-3p-2f/maser_lbf_4s-11x11-3p-2f_results/sacred/maser/lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/lbf_4s-11x11-3p-2f/qmix_lbf_4s-11x11-3p-2f_results/sacred/qmix/lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/lbf_4s-11x11-3p-2f/qplex_lbf_4s-11x11-3p-2f_results/sacred/qplex/lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/lbf_4s-11x11-3p-2f/happo_lbf_4s-11x11-3p-2f_w_parallel_10_threads_results/sacred/happo/lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/lbf_4s-11x11-3p-2f/coma_4s-11x11-3p-2f_w_parallel_10_threads_results/Foraging-4s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/lbf_4s-11x11-3p-2f/eoi_lbf_4s-11x11-3p-2f_w_episode_results/Foraging-4s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/lbf_4s-11x11-3p-2f/mat_dec_lbf_4s-11x11-3p-2f_w_parallel_10_threads_results/sacred/mat_dec/lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/lbf_4s-11x11-3p-2f/mappo_lbf_4s-11x11-3p-2f_w_parallel_10_threads_result/Foraging-4s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/lbf_4s-11x11-3p-2f/maa2c_lbf_4s-11x11-3p-2f_w_parallel_10_threads_results/Foraging-4s-11x11-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/lbf_4s-11x11-3p-2f/emc_lbf_4s-11x11-3p-2f_w_rew_stand_results/sacred/emc/lbforaging:Foraging-4s-11x11-3p-2f-coop-v2"
    # ]
    # algo_names_ = ["CDS", "MASER", "QMIX", "QPLEX", "HAPPO", "COMA", "EOI", "MAT-DEC", "MAPPO", "MAA2C", "EMC"]
    # env_name_ = "lbf_4s-11x11-3p-2f"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/lbf_4s-11x11-3p-2f/"

    # LBF - 2s-9x9-3p-2f
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/lbf_2s-9x9-3p-2f/cds_lbf_2s-9x9-3p-2f_results/sacred/cds/lbforaging:Foraging-2s-9x9-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/lbf_2s-9x9-3p-2f/maser_lbf_2s-9x9-3p-2f_results/sacred/maser/lbforaging:Foraging-2s-9x9-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/lbf_2s-9x9-3p-2f/qplex_lbf_2s-9x9-3p-2f_results/sacred/qplex/lbforaging:Foraging-2s-9x9-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/lbf_2s-9x9-3p-2f/happo_lbf_2s-9x9-3p-2f_w_parallel_10_threads_results/sacred/happo/lbforaging:Foraging-2s-9x9-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/lbf_2s-9x9-3p-2f/coma_lbf_2s-9x9-3p-2f_w_parallel_10_threads_results/Foraging-2s-9x9-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/lbf_2s-9x9-3p-2f/eoi_lbf_2s-9x9-3p-2f_w_episode_results/Foraging-2s-9x9-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/lbf_2s-9x9-3p-2f/qmix_lbf_2s-9x9-3p-2f_results/Foraging-2s-9x9-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/lbf_2s-9x9-3p-2f/mappo_lbf_2s-9x9-3p-2f_w_parallel_10_threads_results/Foraging-2s-9x9-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/lbf_2s-9x9-3p-2f/maa2c_lbf_2s-9x9-3p-2f_w_parallel_10_threads_results/Foraging-2s-9x9-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/lbf_2s-9x9-3p-2f/mat_dec_lbf_2s-9x9-3p-2f_w_parallel_10_threads_results/sacred/mat_dec/lbforaging:Foraging-2s-9x9-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/lbf_2s-9x9-3p-2f/emc_lbf_2s-9x9-3p-2f_w_rew_stand_results/sacred/emc/lbforaging:Foraging-2s-9x9-3p-2f-coop-v2"
    # ]
    # algo_names_ = ["CDS", "MASER", "QPLEX", "HAPPO", "COMA", "EOI", "QMIX", "MAPPO", "MAA2C", "MAT-DEC", "EMC"]
    # env_name_ = "lbf_2s-9x9-3p-2f"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/lbf_2s-9x9-3p-2f/"

    # LBF - 2s-12x12-2p-2f
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/lbf_2s-12x12-2p-2f/cds_lbf_2s-12x12-2p-2f_results/sacred/cds/lbforaging:Foraging-2s-12x12-2p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/lbf_2s-12x12-2p-2f/maser_lbf_2s-12x12-2p-2f_results/sacred/maser/lbforaging:Foraging-2s-12x12-2p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/lbf_2s-12x12-2p-2f/qplex_lbf_2s-12x12-2p-2f_results/sacred/qplex/lbforaging:Foraging-2s-12x12-2p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/lbf_2s-12x12-2p-2f/coma_lbf_2s-12x12-2p-2f_w_parallel_10_threads_results/Foraging-2s-12x12-2p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/lbf_2s-12x12-2p-2f/emc_lbf_2s-12x12-2p-2f_w_rew_stand_results/Foraging-2s-12x12-2p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/lbf_2s-12x12-2p-2f/eoi_lbf_2s-12x12-2p-2f_w_episode_results/Foraging-2s-12x12-2p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/lbf_2s-12x12-2p-2f/happo_lbf_2s-12x12-2p-2f_w_parallel_10_threads_results/sacred/happo/lbforaging:Foraging-2s-12x12-2p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/lbf_2s-12x12-2p-2f/qmix_lbf_2s-12x12-2p-2f_results/Foraging-2s-12x12-2p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/lbf_2s-12x12-2p-2f/mappo_lbf_2s-12x12-2p-2f_w_parallel_10_threads_result/Foraging-2s-12x12-2p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/lbf_2s-12x12-2p-2f/maa2c_lbf_2s-12x12-2p-2f_w_parallel_10_threads_results/Foraging-2s-12x12-2p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/lbf_2s-12x12-2p-2f/mat_dec_lbf_2s-12x12-2p-2f_w_parallel_10_threads_results/sacred/mat_dec/lbforaging:Foraging-2s-12x12-2p-2f-coop-v2",
    # ]
    # algo_names_ = ["CDS", "MASER", "QPLEX", "COMA", "EMC", "EOI", "HAPPO", "QMIX", "MAPPO", "MAA2C", "MAT-DEC"]
    # env_name_ = "2s-12x12-2p-2f"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/lbf_2s-12x12-2p-2f/"

    # LBF - 2s-8x8-3p-2f
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/lbf_2s-8x8-3p-2f/cds_lbf_2s-8x8-3p-2f_results/sacred/cds/lbforaging:Foraging-2s-8x8-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/lbf_2s-8x8-3p-2f/maser_lbf_2s-8x8-3p-2f_results/sacred/maser/lbforaging:Foraging-2s-8x8-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/lbf_2s-8x8-3p-2f/qplex_lbf_2s-8x8-3p-2f_results/sacred/qplex/lbforaging:Foraging-2s-8x8-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/lbf_2s-8x8-3p-2f/coma_lbf_2s-8x8-3p-2f_w_parallel_10_threads_results",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/lbf_2s-8x8-3p-2f/emc_lbf_2s-8x8-3p-2f_w_rew_stand_results/Foraging-2s-8x8-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/lbf_2s-8x8-3p-2f/eoi_lbf_2s-8x8-3p-2f_w_episode_results/Foraging-2s-8x8-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/lbf_2s-8x8-3p-2f/mat_dec_lbf_2s-8x8-3p-2f_w_parallel_10_threads_results/sacred/mat_dec/lbforaging:Foraging-2s-8x8-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/lbf_2s-8x8-3p-2f/qmix_lbf_2s-8x8-3p-2f_results/Foraging-2s-8x8-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/lbf_2s-8x8-3p-2f/mappo_lbf_2s-8x8-3p-2f_w_parallel_10_threads_results/Foraging-2s-8x8-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/lbf_2s-8x8-3p-2f/maa2c_lbf_2s-8x8-3p-2f_w_parallel_10_threads_results/Foraging-2s-8x8-3p-2f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/lbf_2s-8x8-3p-2f/happo_lbf_2s-8x8-3p-2f_w_parallel_10_threads_results/sacred/happo/lbforaging:Foraging-2s-8x8-3p-2f-coop-v2",
    # ]
    # algo_names_ = ["CDS", "MASER", "QPLEX", "COMA", "EMC", "EOI", "MAT-DEC", "QMIX", "MAPPO", "MAA2C", "HAPPO"]
    # env_name_ = "2s-8x8-3p-2f"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/lbf_2s-8x8-3p-2f/"

    # LBF - 7s-20x20-5p-3f
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/lbf_7s-20x20-5p-3f/cds_lbf_7s-20x20-5p-3f_results/sacred/cds/lbforaging:Foraging-7s-20x20-5p-3f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/lbf_7s-20x20-5p-3f/qplex_lbf_7s-20x20-5p-3f_results/sacred/qplex/lbforaging:Foraging-7s-20x20-5p-3f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/lbf_7s-20x20-5p-3f/maser_lbf_7s-20x20-5p-3f_results/sacred/maser/lbforaging:Foraging-7s-20x20-5p-3f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/lbf_7s-20x20-5p-3f/happo_lbf_7s-20x20-5p-3f_w_parallel_10_threads_results/sacred/happo/lbforaging:Foraging-7s-20x20-5p-3f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/lbf_7s-20x20-5p-3f/coma_lbf_7s-20x20-5p-3f_w_parallel_10_threads_results/Foraging-7s-20x20-5p-3f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/lbf_7s-20x20-5p-3f/emc_lbf_7s-20x20-5p-3f_w_rew_stand_results/Foraging-7s-20x20-5p-3f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/lbf_7s-20x20-5p-3f/eoi_lbf_7s-20x20-5p-3f_w_episode_results/Foraging-7s-20x20-5p-3f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/lbf_7s-20x20-5p-3f/qmix_lbf_7s-20x20-5p-3f_results/Foraging-7s-20x20-5p-3f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/lbf_7s-20x20-5p-3f/mappo_lbf_7s-20x20-5p-3f_w_parallel_10_threads_result/Foraging-7s-20x20-5p-3f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/lbf_7s-20x20-5p-3f/maa2c_lbf_7s-20x20-5p-3f_w_parallel_10_threads_results/Foraging-7s-20x20-5p-3f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/lbf_7s-20x20-5p-3f/mat_dec_lbf_7s-20x20-5p-3f_w_parallel_10_threads_results/sacred/mat_dec/lbforaging:Foraging-7s-20x20-5p-3f-coop-v2"
    # ]
    # algo_names_ = ["CDS", "QPLEX", "MASER", "HAPPO", "COMA", "EMC", "EOI", "QMIX", "MAPPO", "MAA2C", "MAT-DEC"]
    # env_name_ = "7s-20x20-5p-3f"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/lbf_7s-20x20-5p-3f/"

    # LBF - 8s-25x25-8p-5f
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/lbf_8s-25x25-8p-5f/cds_lbf_8s-25x25-8p-5f_results/sacred/cds/lbforaging:Foraging-8s-25x25-8p-5f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/lbf_8s-25x25-8p-5f/qplex_lbf_8s-25x25-8p-5f_results/sacred/qplex/lbforaging:Foraging-8s-25x25-8p-5f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/lbf_8s-25x25-8p-5f/maser_lbf_8s-25x25-8p-5f_results/sacred/maser/lbforaging:Foraging-8s-25x25-8p-5f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/lbf_8s-25x25-8p-5f/coma_lbf_8s-25x25-8p-5f_w_parallel_10_threads_results/Foraging-8s-25x25-8p-5f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/lbf_8s-25x25-8p-5f/eoi_lbf_8s-25x25-8p-5f_w_episode_results/Foraging-8s-25x25-8p-5f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/lbf_8s-25x25-8p-5f/qmix_lbf_8s-25x25-8p-5f_results/Foraging-8s-25x25-8p-5f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/lbf_8s-25x25-8p-5f/mappo_lbf_8s-25x25-8p-5f_w_parallel_10_threads_result/Foraging-8s-25x25-8p-5f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/lbf_8s-25x25-8p-5f/maa2c_lbf_8s-25x25-8p-5f_w_parallel_10_threads_results/Foraging-8s-25x25-8p-5f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/lbf_8s-25x25-8p-5f/mat_dec_lbf_8s-25x25-8p-5f_w_parallel_10_threads_results/sacred/mat_dec/lbforaging:Foraging-8s-25x25-8p-5f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/lbf_8s-25x25-8p-5f/happo_lbf_8s-25x25-8p-5f_w_parallel_10_threads_results/sacred/happo/lbforaging:Foraging-8s-25x25-8p-5f-coop-v2"
    # ]
    # algo_names_ = ["CDS", "QPLEX", "MASER", "COMA", "EOI", "QMIX", "MAPPO", "MAA2C", "MAT-DEC", "HAPPO"]
    # env_name_ = "8s-25x25-8p-5f"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/lbf_8s-25x25-8p-5f/"

    # LBF - 7s-30x30-7p-4f
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/lbf_7s-30x30-7p-4f/cds_lbf_7s-30x30-7p-4f_results/sacred/cds/lbforaging:Foraging-7s-30x30-7p-4f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/lbf_7s-30x30-7p-4f/qplex_lbf_7s-30x30-7p-4f_results/sacred/qplex/lbforaging:Foraging-7s-30x30-7p-4f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/lbf_7s-30x30-7p-4f/maser_lbf_7s-30x30-7p-4f_results/sacred/maser/lbforaging:Foraging-7s-30x30-7p-4f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/lbf_7s-30x30-7p-4f/happo_lbf_7s-30x30-7p-4f_w_parallel_10_threads_results/sacred/happo/lbforaging:Foraging-7s-30x30-7p-4f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/lbf_7s-30x30-7p-4f/coma_lbf_7s-30x30-7p-4f_w_parallel_10_threads_results/Foraging-7s-30x30-7p-4f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/lbf_7s-30x30-7p-4f/eoi_lbf_7s-30x30-7p-4f_w_episode_results/Foraging-7s-30x30-7p-4f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/lbf_7s-30x30-7p-4f/qmix_lbf_7s-30x30-7p-4f_results/Foraging-7s-30x30-7p-4f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/lbf_7s-30x30-7p-4f/mappo_lbf_7s-30x30-7p-4f_w_parallel_10_threads_result/Foraging-7s-30x30-7p-4f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/lbf_7s-30x30-7p-4f/maa2c_lbf_7s-30x30-7p-4f_w_parallel_10_threads_results/Foraging-7s-30x30-7p-4f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/lbf_7s-30x30-7p-4f/emc_lbf_7s-30x30-7p-4f_wo_rew_stand_results/sacred/emc/lbforaging:Foraging-7s-30x30-7p-4f-coop-v2",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/lbf_7s-30x30-7p-4f/mat_dec_lbf_7s-30x30-7p-4f_w_parallel_10_threads_results/sacred/mat_dec/lbforaging:Foraging-7s-30x30-7p-4f-coop-v2"
    # ]
    # algo_names_ = ["CDS", "QPLEX", "MASER", "HAPPO", "COMA", "EOI", "QMIX", "MAPPO", "MAA2C", "EMC", "MAT-DEC"]
    # env_name_ = "7s-30x30-7p-4f"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/lbf_7s-30x30-7p-4f/"

    # RWARE - tiny-4ag-hard-v1
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/rware-tiny-4ag-hard-v1/cds_rware-tiny-4ag-hard-v1_results/sacred/cds/rware:rware-tiny-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/rware-tiny-4ag-hard-v1/qmix_rware-tiny-4ag-hard-v1_results/sacred/qmix/rware:rware-tiny-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/rware-tiny-4ag-hard-v1/qplex_rware-tiny-4ag-hard-v1_results/sacred/qplex/rware-tiny-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/rware-tiny-4ag-hard-v1/happo_rware-tiny-4ag-hard_w_parallel_10_threads_results/sacred/happo/rware:rware-tiny-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/rware-tiny-4ag-hard-v1/coma_rware_rware-tiny-4ag-hard-v1_w_parallel_10_threads_results/rware_rware-tiny-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/rware-tiny-4ag-hard-v1/emc_rware_rware-tiny-4ag-hard-v1_w_rew_stand_results/rware_rware-tiny-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/rware-tiny-4ag-hard-v1/eoi_rware_rware-tiny-4ag-hard-v1_w_episode_results/rware_rware-tiny-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/rware-tiny-4ag-hard-v1/mat_dec_rware_rware-tiny-4ag-hard_w_parallel_10_threads_results/sacred/mat_dec/rware:rware-tiny-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/rware-tiny-4ag-hard-v1/mappo_rware_rware-tiny-4ag-hard-v1_w_parallel_10_threads_results/rware_rware-tiny-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/rware-tiny-4ag-hard-v1/maa2c_rware_rware-tiny-4ag-hard-v1_w_parallel_10_threads_results/rware_rware-tiny-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/rware-tiny-4ag-hard-v1/maser_rware-tiny-4ag-hard-v1_results/sacred/maser/rware:rware-tiny-4ag-hard-v1"
    # ]
    # algo_names_ = ["CDS", "QMIX", "QPLEX", "HAPPO", "COMA", "EMC", "EOI", "MAT-DEC", "MAPPO", "MAA2C", "MASER"]
    # env_name_ = "tiny-4ag-hard-v1"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/rware_tiny-4ag-hard-v1/"

    # RWARE - tiny-2ag-hard-v1
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/rware-tiny-2ag-hard-v1/emc_rware_rware-tiny-2ag-hard-v1_w_rew_stand_results/sacred/emc/rware:rware-tiny-2ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/rware-tiny-2ag-hard-v1/qmix_rware-tiny-2ag-hard-v1_results/sacred/qmix/rware:rware-tiny-2ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/rware-tiny-2ag-hard-v1/qplex_rware-tiny-2ag-hard-v1_results/sacred/qplex/rware-tiny-2ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/rware-tiny-2ag-hard-v1/cds_rware-tiny-2ag-hard-v1_results/sacred/cds/rware:rware-tiny-2ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/rware-tiny-2ag-hard-v1/happo_rware-tiny-2ag-hard_w_parallel_10_threads_results/sacred/happo/rware:rware-tiny-2ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/rware-tiny-2ag-hard-v1/coma_rware_rware-tiny-2ag-hard-v1_w_parallel_10_threads_results/rware_rware-tiny-2ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/rware-tiny-2ag-hard-v1/eoi_rware_rware-tiny-2ag-hard-v1_w_episode_results/rware_rware-tiny-2ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/rware-tiny-2ag-hard-v1/mappo_rware_rware-tiny-2ag-hard-v1_w_parallel_10_threads_results/rware_rware-tiny-2ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/rware-tiny-2ag-hard-v1/maa2c_rware_rware-tiny-2ag-hard-v1_w_parallel_10_threads_results/rware_rware-tiny-2ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/rware-tiny-2ag-hard-v1/maser_rware-tiny-2ag-hard-v1_results/sacred/maser/rware:rware-tiny-2ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/rware-tiny-2ag-hard-v1/mat_dec_rware_rware-tiny-2ag-hard-v1_w_parallel_10_threads_results/sacred/mat_dec/rware:rware-tiny-2ag-hard-v1"
    # ]
    # algo_names_ = ["EMC", "QMIX", "QPLEX", "CDS", "HAPPO", "COMA", "EOI", "MAPPO", "MAA2C", "MASER", "MAT-DEC"]
    # env_name_ = "tiny-2ag-hard-v1"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/rware_tiny-2ag-hard-v1/"

    # RWARE - small-4ag-hard-v1
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/rware-small-4ag-hard-v1/emc_rware-small-4ag-hard-v1_results/sacred/emc/rware:rware-small-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/rware-small-4ag-hard-v1/qmix_rware-small-4ag-hard-v1_results/sacred/qmix/rware:rware-small-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/rware-small-4ag-hard-v1/qplex_rware-small-4ag-hard-v1_results/sacred/qplex/rware_rware-small-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/rware-small-4ag-hard-v1/cds_rware-small-4ag-hard-v1_results/sacred/cds/rware:rware-small-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/rware-small-4ag-hard-v1/coma_rware_rware-small-4ag-hard-v1_w_parallel_10_threads_results/rware_rware-small-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/rware-small-4ag-hard-v1/eoi_rware_rware-small-4ag-hard-v1_w_episode_results/rware_rware-small-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/rware-small-4ag-hard-v1/happo_rware-small-4ag-hard_w_parallel_10_threads_results/sacred/happo/rware:rware-small-4ag-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/rware-small-4ag-hard-v1/mappo_rware_rware-small-4ag-hard-v1_w_parallel_10_threads_results/rware_rware-small-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/rware-small-4ag-hard-v1/maa2c_rware_rware-small-4ag-hard-v1_w_parallel_10_threads_results/rware_rware-small-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/rware-small-4ag-hard-v1/maser_rware-small-4ag-hard-v1_results/sacred/maser/rware:rware-small-4ag-hard-v1",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/rware-small-4ag-hard-v1/mat_dec_rware_rware-small-4ag-hard-v1_w_parallel_10_threads_results/sacred/mat_dec/rware:rware-small-4ag-hard-v1",
    # ]
    # algo_names_ = ["EMC", "QMIX", "QPLEX", "CDS", "COMA", "EOI", "HAPPO", "MAPPO", "MAA2C", "MASER", "MAT-DEC"]
    # env_name_ = "small-4ag-hard-v1"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/rware_small-4ag-hard-v1/"

    # MPE - SimpleSpread-3
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/mpe_SimpleSpread-3/qplex_mpe_SimpleSpread-3_results/sacred/qplex/mpe:SimpleSpread-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/mpe_SimpleSpread-3/maser_mpe_SimpleSpread-3_results/sacred/maser/mpe:SimpleSpread-3-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/mpe_SimpleSpead-3/eoi_mpe_SimpleSpead-3_w_episode_results/sacred/eoi/mpe:SimpleSpread-3-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/mpe_SimpleSpread-3/coma_mpe_SimpleSpread-3_w_parallel_10_threads_results/mpe_SimpleSpread-3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/mpe_SimpleSpread-3/mappo_mpe_SimpleSpread-3_w_parallel_10_threads_results/spread-3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/mpe_SimpleSpread-3/maa2c_mpe_SimpleSpread-3_w_parallel_10_threads_results/spread-3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/mpe_SimpleSpread-3/happo_mpe_SimpleSpread-3_w_parallel_10_threads_results/sacred/happo/mpe:SimpleSpread-3-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/mpe_SimpleSpread-3/mat_dec_mpe_SimpleSpread-3_w_parallel_10_threads_results/sacred/mat_dec/mpe:SimpleSpread-3-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/mpe_SimpleSpread-3/cds_mpe_SimpleSpread-3_results/sacred/cds/mpe:SimpleSpread-3-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/mpe_SimpleSpread-3/emc_mpe_SimpleSpread-3_results/sacred/emc/mpe:SimpleSpread-3-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/mpe_SimpleSpread-3/qmix_mpe_SimpleSpread-3_results/sacred/qmix/mpe:SimpleSpread-3-v0"
    # ]
    # algo_names_ = ["QPLEX", "MASER", "EOI", "COMA", "MAPPO", "MAA2C", "HAPPO", "MAT-DEC", "CDS", "EMC", "QMIX"]
    # env_name_ = "SimpleSpread-3"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/simplespread-3/"

    # MPE - SimpleSpread-4
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/mpe_SimpleSpread-4/emc_mpe_SimpleSpread-4_results/sacred/emc/mpe:SimpleSpread-4-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/mpe_SimpleSpread-4/qplex_mpe_SimpleSpread-4_results/sacred/qplex/mpe:SimpleSpread-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/mpe_SimpleSpead-4/eoi_mpe_SimpleSpead-4_w_episode_results/sacred/eoi/mpe:SimpleSpread-4-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/mpe_SimpleSpread-4/maser_mpe_SimpleSpread-4_results/sacred/maser/mpe:SimpleSpread-4-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/mpe_SimpleSpread-4/cds_mpe_SimpleSpread-4_results/sacred/cds/mpe:SimpleSpread-4-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/mpe_SimpleSpread-4/coma_mpe_SimpleSpread-4_w_parallel_10_threads_results/mpe_SimpleSpread-4",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/mpe_SimpleSpread-4/qmix_mpe_SimpleSpread-4_results/spread-4",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/mpe_SimpleSpread-4/mappo_mpe_SimpleSpread-4_w_parallel_10_threads_results/spread-4",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/mpe_SimpleSpread-4/maa2c_mpe_SimpleSpread-4_w_parallel_10_threads_results/spread-4",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/mpe_SimpleSpread-4/happo_mpe_SimpleSpread-4_w_parallel_10_threads_results/sacred/happo/mpe:SimpleSpread-4-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/mpe_SimpleSpread-4/mat_dec_mpe_SimpleSpread-4_w_parallel_10_threads_results/sacred/mat_dec/mpe:SimpleSpread-4-v0",
    # ]
    # algo_names_ = ["EMC", "QPLEX", "EOI", "MASER", "CDS", "COMA", "QMIX", "MAPPO", "MAA2C", "HAPPO", "MAT-DEC"]
    # env_name_ = "SimpleSpread-4"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/simplespread-4/"

    # MPE - SimpleSpread-5
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/mpe_SimpleSpread-5/emc_mpe_SimpleSpread-5_results/sacred/emc/mpe:SimpleSpread-5-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/mpe_SimpleSpread-5/qplex_mpe_SimpleSpread-5_results/sacred/qplex/mpe:SimpleSpread-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/mpe_SimpleSpread-5/coma_mpe_SimpleSpread-5_w_parallel_10_threads_results/mpe_SimpleSpread-5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/mpe_SimpleSpread-5/qmix_mpe_SimpleSpread-5_results/spread-5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/mpe_SimpleSpread-5/mappo_mpe_SimpleSpread-5_w_parallel_10_threads_results/spread-5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/mpe_SimpleSpread-5/maa2c_mpe_SimpleSpread-5_w_parallel_10_threads_results/spread-5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/mpe_SimpleSpread-5/happo_mpe_SimpleSpread-5_w_parallel_10_threads_results/sacred/happo/mpe:SimpleSpread-5-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/mpe_SimpleSpread-5/mat_dec_mpe_SimpleSpread-5_w_parallel_10_threads_results/sacred/mat_dec/mpe:SimpleSpread-5-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/mpe_SimpleSpread-5/cds_mpe_SimpleSpread-5_results/sacred/cds/mpe:SimpleSpread-5-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/mpe_SimpleSpead-5/eoi_mpe_SimpleSpead-5_w_episode_results/sacred/eoi/mpe:SimpleSpread-5-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/mpe_SimpleSpread-5/maser_mpe_SimpleSpread-5_results/sacred/maser/mpe:SimpleSpread-5-v0",
    # ]
    # algo_names_ = ["EMC", "QPLEX", "COMA", "QMIX", "MAPPO", "MAA2C", "HAPPO", "MAT-DEC", "CDS", "EOI", "MASER"]
    # env_name_ = "SimpleSpread-5"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/simplespread-5/"

    # MPE - SimpleSpread-8
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/mpe_SimpleSpread-8/maser_mpe_SimpleSpread-8_results/sacred/maser/mpe:SimpleSpread-8-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/mpe_SimpleSpread-8/qplex_mpe_SimpleSpread-8_results/sacred/qplex/mpe:SimpleSpread-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/mpe_SimpleSpread-8/coma_mpe_SimpleSpread-8_w_parallel_10_threads_results/mpe_SimpleSpread-8",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/mpe_SimpleSpread-8/qmix_mpe_SimpleSpread-8_results/spread-8",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/mpe_SimpleSpread-8/mappo_mpe_SimpleSpread-8_w_parallel_10_threads_results/spread-8",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/mpe_SimpleSpread-8/maa2c_mpe_SimpleSpread-8_w_parallel_10_threads_results/spread-8",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/mpe_SimpleSpread-8/happo_mpe_SimpleSpread-8_w_parallel_10_threads_results/sacred/happo/mpe:SimpleSpread-8-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/mpe_SimpleSpread-8/mat_dec_mpe_SimpleSpread-8_w_parallel_10_threads_results/sacred/mat_dec/mpe:SimpleSpread-8-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/mpe_SimpleSpread-8/cds_mpe_SimpleSpread-8_results/sacred/cds/mpe:SimpleSpread-8-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/mpe_SimpleSpead-8/eoi_mpe_SimpleSpead-8_w_episode_results/sacred/eoi/mpe:SimpleSpread-8-v0",
    # ]
    # algo_names_ = ["MASER", "QPLEX", "COMA", "QMIX", "MAPPO", "MAA2C", "HAPPO", "MAT-DEC", "CDS", "EOI"]
    # env_name_ = "SimpleSpread-8"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/simplespread-8/"

    # MPE - SimpleSpeakerListener
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/mpe_SimpleSpeakerListener/eoi_mpe_SimpleSpeakerListener_w_episode_results/sacred/eoi/mpe:SimpleSpeakerListener-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/mpe_SimpleSpeakerListener/qplex_mpe_SimpleSpeakerListener_results/results/sacred/qplex/mpe:SimpleSpeakerListener-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/mpe_SimpleSpeakerListener/qmix_mpe_SimpleSpeakerListener_results/sacred/qmix/mpe:SimpleSpeakerListener-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/CDS/mpe_SimpleSpeakerListener/cds_mpe_SimpleSpeakerListener_results/sacred/cds/mpe:SimpleSpeakerListener-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/mpe_SimpleSpeakerListener/coma_mpe_SimpleSpeakerListener_w_parallel_10_threads_results/mpe_SimpleSpeakerListener-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/mpe_SimpleSpeakerListener/mappo_mpe_SimpleSpeakerListener_w_parallel_10_threads_results/mpe_SimpleSpeakerListener-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/mpe_SimpleSpeakerListener/maa2c_mpe_SimpleSpeakerListener_w_parallel_10_threads_results/mpe_SimpleSpeakerListener-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/HAPPO/mpe_SimpleSpeakerListener/happo_mpe_SimpleSpeakerListener_w_parallel_10_threads_results/sacred/happo/mpe:SimpleSpeakerListener-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAT_DEC/mpe_SimpleSpeakerListener/mat_dec_mpe_SimpleSpeakerListener_w_parallel_10_threads_results/sacred/mat_dec/mpe:SimpleSpeakerListener-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MASER/mpe_SimpleSpeakerListener/maser_mpe_SimpleSpeakerListener_results/sacred/maser/mpe:SimpleSpeakerListener-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EMC/mpe_SimpleSpeakerListener/emc_mpe_SimpleSpeakerListener_results/sacred/emc/mpe:SimpleSpeakerListener-v0"
    # ]
    # algo_names_ = ["EOI", "QPLEX", "QMIX", "CDS", "COMA", "MAPPO", "MAA2C", "HAPPO", "MAT-DEC", "MASER", "EMC"]
    # env_name_ = "SimpleSpeakerListener"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/simplespeakerlistener/"

    # plot_train_ = False
    # plot_legend_bool_ = False
    # plot_multiple_experiment_results(
    #     paths_to_results_,
    #     algo_names_,
    #     env_name_,
    #     path_to_save_,
    #     plot_train_,
    #     plot_legend_bool_
    # )

    ## Average plots per algo for all tasks of a benchmark

    # PettingZoo
    # _paths_to_pickle_results = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/pistonball_v6/all_results_env=pistonball_v6.pkl",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/cooperative_pong_v5/all_results_env=cooperative_pong_v5.pkl",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/entombed_cooperative_v3/all_results_env=entombed_cooperative_v3.pkl"
    # ]
    # _plot_title = "PettingZoo"
    # _path_to_save = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/pettingzoo/"

    # Pressure Plate
    # _paths_to_pickle_results = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/pressureplate-linear-4p-v0/all_results_env=pressureplate-linear-4p-v0.pkl",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/pressureplate-linear-6p-v0/all_results_env=pressureplate-linear-6p-v0.pkl",
    # ]
    # _plot_title = "Pressure Plate"
    # _path_to_save = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/pressureplate/"

    # MPE
    # _paths_to_pickle_results = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/simplespread-4/all_results_env=SimpleSpread-4.pkl",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/simplespread-5/all_results_env=SimpleSpread-5.pkl",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/simplespread-8/all_results_env=SimpleSpread-8.pkl"
    # ]
    # _plot_title = "MPE"
    # _path_to_save = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/MPE/"

    # RWARE
    # _paths_to_pickle_results = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/rware_tiny-2ag-hard-v1/all_results_env=tiny-2ag-hard-v1.pkl",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/rware_tiny-4ag-hard-v1/all_results_env=tiny-4ag-hard-v1.pkl",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/rware_small-4ag-hard-v1/all_results_env=small-4ag-hard-v1.pkl"
    # ]
    # _plot_title = "RWARE"
    # _path_to_save = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/RWARE/"

    # LBF
    # _paths_to_pickle_results = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/lbf_2s-8x8-3p-2f/all_results_env=2s-8x8-3p-2f.pkl",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/lbf_2s-9x9-3p-2f/all_results_env=lbf_2s-9x9-3p-2f.pkl",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/lbf_2s-12x12-2p-2f/all_results_env=2s-12x12-2p-2f.pkl",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/lbf_4s-11x11-3p-2f/all_results_env=lbf_4s-11x11-3p-2f.pkl",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/lbf_7s-20x20-5p-3f/all_results_env=7s-20x20-5p-3f.pkl",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/lbf_7s-30x30-7p-4f/all_results_env=7s-30x30-7p-4f.pkl",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/lbf_8s-25x25-8p-5f/all_results_env=8s-25x25-8p-5f.pkl"
    # ]
    # _plot_title = "LBF"
    # _path_to_save = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/LBF/"

    # Overcooked
    # _paths_to_pickle_results = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/cramped_room/all_results_env=Cramped Room.pkl",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/assymetric_advantages/all_results_env=Asymmetric Advantages.pkl",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/coordination_ring/all_results_env=Coordination Ring.pkl"
    # ]
    # _plot_title = "Overcooked"
    # _path_to_save = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/Overcooked/"

    # _plot_legend = False
    # plot_average_per_algo_for_all_tasks_of_a_benchmark(
    #     _paths_to_pickle_results,
    #     _plot_title,
    #     _path_to_save,
    #     _plot_legend
    # )

    # Create just a legend
    _path_to_save = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/"
    create_only_legend(_path_to_save)
