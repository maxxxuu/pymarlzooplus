import json
import os

from matplotlib import pyplot as plt
import numpy as np


def read_json(json_path):

    assert os.path.exists(json_path), \
        f"The provided path to json file does not exist! \n'json_path': {json_path}"

    # Open the file for reading
    with open(json_path, 'r') as file:
        # Load data from the file into a Python dictionary
        data = json.load(file)

    return data


def create_plot(x_data,
                y_data_mean,
                y_data_std,
                path_to_save,
                x_label,
                y_label,
                plot_title,
                legend_labels=None):

    assert len(x_data) == len(y_data_mean), \
        f"'len(x_data)': {len(x_data)}, 'len(y_data_mean)': {len(y_data_mean)}"
    if y_data_std[0] is not None:
        assert len(y_data_std) == len(y_data_mean), \
            f"'len(y_data_std)': {len(y_data_std)}, 'len(y_data_mean)': {len(y_data_mean)}"

    # Create new figure
    plt.figure()

    for data_idx in range(len(x_data)):
        # Plot the data
        plt.plot(x_data[data_idx],
                 y_data_mean[data_idx],
                 label=None if legend_labels is None else legend_labels[data_idx])

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

    return (x_return_data,
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
            (x_return_data,
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
            create_plot([x_return_data, x_test_return_data],
                        [return_mean_data, test_return_mean_data],
                        [return_std_data, test_return_std_data],
                        path_to_save,
                        "timesteps",
                        "return mean",
                        plot_title,
                        legend_labels=["Train", "Test"]
                        )

            # Create plot for the normalized returns
            path_to_save = os.path.join(path_to_save_results, "normalized_return_mean")
            create_plot([x_return_data, x_test_return_data],
                        [normalized_return_mean_data, test_normalized_return_mean_data],
                        [normalized_return_std_data, test_normalized_return_std_data],
                        path_to_save,
                        "timesteps",
                        "per-step return mean",
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
            create_plot([x_data],
                        [mean_data],
                        [std_data],
                        path_to_save,
                        "timesteps",
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
    truncated_data = [truncate_data(np.array(x), np.array(y), max_common_time)
                      for x, y in zip(x_data_list, mean_data_list)]

    # Step 3: Define a common set of timesteps and interpolate
    # Increase the data resolution by a factor of 10.
    common_timeline = np.linspace(1, max_common_time, num=len(truncated_data[0][0])*10)
    interpolated_data = np.array([np.interp(common_timeline, x, y) for x, y in truncated_data])

    # Step 4: Calculate mean and standard deviation
    mean_data = np.mean(interpolated_data, axis=0)
    std_data = np.std(interpolated_data, axis=0)

    return mean_data, std_data, common_timeline


def create_multiple_exps_plot(all_results,
                              path_to_save,
                              plot_title,
                              legend_labels,  # Algorithm names
                              env_name,
                              plot_train=True):

    # Create new figures, one for returns and another one for per-set returns.
    plt.figure(1)
    plt.xlabel("Timesteps")
    plt.ylabel("return mean")
    plt.title(plot_title)

    plt.figure(2)
    plt.xlabel("Timesteps")
    plt.ylabel("per-step return mean")
    plt.title(plot_title)

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

        data_for_plots = [[mean_data,
                           std_data,
                           common_timeline,
                           test_mean_data,
                           test_std_data,
                           test_common_timeline
                           ],
                          [norm_mean_data,
                           norm_std_data,
                           common_timeline,
                           norm_test_mean_data,
                           norm_test_std_data,
                           test_common_timeline
                           ]
                          ]

        for data_for_plot_idx, data_for_plot in enumerate(data_for_plots):

            # Set which figure to update
            plt.figure(data_for_plot_idx+1)

            # Plot the test data
            plot_legend = legend_labels[alg_idx] if plot_train is False else legend_labels[alg_idx] + "-test"
            plt.plot(data_for_plot[5],
                     data_for_plot[3],
                     label=plot_legend
                     )

            # Add std if available
            if data_for_plot[4] is not None:

                # Calculate the upper and lower bounds of the standard deviation
                std_upper = np.array(data_for_plot[3]) + 1.15*np.array(data_for_plot[4])  # 75%
                std_lower = np.array(data_for_plot[3]) - 1.15*np.array(data_for_plot[4])  # 75%

                # Add a shaded area for the standard deviation
                plt.fill_between(data_for_plot[5], std_lower, std_upper, alpha=0.2)

            # Plot the train data
            if plot_train is True:
                plot_legend = legend_labels[alg_idx] + "-train"
                plt.plot(data_for_plot[2],
                         data_for_plot[0],
                         label=plot_legend
                         )
                # Add std if available
                if data_for_plot[1] is not None:
                    # Calculate the upper and lower bounds of the standard deviation
                    std_upper = np.array(data_for_plot[0]) + np.array(data_for_plot[1])
                    std_lower = np.array(data_for_plot[0]) - np.array(data_for_plot[1])
                    # Add a shaded area for the standard deviation
                    plt.fill_between(data_for_plot[2], std_lower, std_upper, alpha=0.2)

    # Adding legend, save, and close
    plt.figure(1)
    plt.legend()
    plt.tight_layout()
    path_to_save_plot = os.path.join(path_to_save, f"return_mean_env={env_name}")
    plt.savefig(path_to_save_plot)
    plt.close()

    plt.figure(2)
    plt.legend()
    plt.tight_layout()
    path_to_save_plot = os.path.join(path_to_save, f"normalized_return_mean_env={env_name}")
    plt.savefig(path_to_save_plot)
    plt.close()


def plot_multiple_experiment_results(paths_to_results, algo_names, env_name, path_to_save, plot_train):
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
        # # Check if the order of paths are on par with the order of the algorithm names
        # assert os.path.basename(os.path.abspath(os.path.join(path_to_results, '..'))) == algo_names[path_to_results_idx], \
        #     f"The order of paths should be aligned with the order of algorithm names!"

        path_to_exps = [os.path.join(path_to_results, elem) for elem in os.listdir(path_to_results) if elem.isdigit()]
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
            (x_return_data,
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

        (mean_data,
         std_data,
         common_timeline) = calculate_mean_and_std_of_multiple_exps(x_return_data_list, return_mean_data_list)
        (norm_mean_data,
         norm_std_data,
         _
         ) = calculate_mean_and_std_of_multiple_exps(x_return_data_list, normalized_return_mean_data_list)
        (test_mean_data,
         test_std_data,
         test_common_timeline
         ) = calculate_mean_and_std_of_multiple_exps(x_test_return_data_list, test_return_mean_data_list)
        (norm_test_mean_data,
         norm_test_std_data,
         _
         ) = calculate_mean_and_std_of_multiple_exps(x_test_return_data_list, test_normalized_return_mean_data_list)
        all_results.append([mean_data, std_data, norm_mean_data, norm_std_data, common_timeline,
                            test_mean_data, test_std_data, norm_test_mean_data, norm_test_std_data, test_common_timeline
                            ])

    # Create plots
    plot_title = "Env: " + env_name
    if os.path.exists(path_to_save) is False:
        os.makedirs(path_to_save)
    create_multiple_exps_plot(all_results,
                              path_to_save,
                              plot_title,
                              algo_names,
                              env_name,
                              plot_train=plot_train)

    print("\nMultiple-experiment plots created successfully! "
          f"\nSaved at: {path_to_save}")


if __name__ == '__main__':

    # Single algo
    # path_to_results_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/entombed_cooperative_v3/entombed_cooperative_v3_parallel_2_threads_w_resnet_18_max_steps=20M_hidden_dim=256_results/sacred/mappo/entombed_cooperative_v3/1"
    # algo_name_ = "mappo"
    # env_name_ = "entombed_cooperative_v3"
    # plot_single_experiment_results(path_to_results_, algo_name_, env_name_)

    ## Many algos

    # Pistonball
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/pistonball_v6/coma_pistonball_v6_w_parallel_2_threads_w_resnet18_results/sacred/coma/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/pistonball_v6/coma_pistonball_v6_w_parallel_2_threads_w_trainable_cnn_results/sacred/coma/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/pistonball_v6/pistonball_v6_w_parallel_2_threads_w_resnet18_results/sacred/maa2c/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/pistonball_v6/maa2c_pistonball_v6_w_parallel_2_threads_w_trainable_cnn_results/sacred/maa2c/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/pistonball_v6/mappo_pistonball_v6_w_parallel_2_threads_w_resnet18_results/results/sacred/mappo/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/pistonball_v6/pistonball_v6_w_resnet18_results/results/sacred/qmix/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/pistonball_v6/pistonball_v6_w_trainable_cnn_buffer_size=30_results/results/sacred/qmix/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/pistonball_v6/pistonball_v6_w_episode_w_resnet18_results/sacred/eoi/pistonball_v6",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/pistonball_v6/pistonball_v6_w_resnet18_results/results/sacred/qplex/pistonball_v6"
    #            ]
    # algo_names_ = ["coma-resnet", "coma-cnn", "maa2c-resnet", "maa2c-cnn", "mappo", "qmix-resnet", "qmix-cnn-buff=30", "eoi", "qplex"]
    # env_name_ = "pistonball_v6"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/pistonball_v6/"

    # Cooperative pong
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/cooperative_pong_v5/coma_cooperative_pong_v5_w_parallel_2_threads_w_resnet18_results/sacred/coma/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/cooperative_pong_v5/cooperative_pong_v5_w_parallel_2_threads_w_resnet18_results/sacred/maa2c/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/cooperative_pong_v5/cooperative_pong_v5_w_parallel_5_threads_w_resnet18_results/sacred/maa2c/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/cooperative_pong_v5/cooperative_pong_v5_w_parallel_2_threads_w_trainable_cnn_results/sacred/maa2c/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/cooperative_pong_v5/mappo_cooperative_pong_v5_w_parallel_2_threads_w_resnet18_results/sacred/mappo/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/cooperative_pong_v5/cooperative_pong_v5_w_resnet18_results/sacred/qmix/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/cooperative_pong_v5/eoi_cooperative_pong_v5_w_episode_w_resnet18_results/sacred/eoi/cooperative_pong_v5",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/cooperative_pong_v5/cooperative_pong_v5_w_resnet18_results/sacred/qplex/cooperative_pong_v5"
    #                    ]
    # algo_names_ = ["coma", "maa2c-thr=2", "maa2c-thr=5", "maa2c-cnn-thr=2", "mappo", "qmix", "eoi", "qplex"]
    # env_name_ = "cooperative_pong_v5"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/cooperative_pong_v5/"

    # Entombed cooperative
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/entombed_cooperative_v3/coma_entombed_cooperative_v3_w_parallel_2_threads_w_resnet18_results/results/sacred/coma/entombed_cooperative_v3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/entombed_cooperative_v3/entombed_cooperative_v3_w_parallel_2_threads_w_resnet18_results/results/sacred/maa2c/entombed_cooperative_v3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/entombed_cooperative_v3/entombed_cooperative_v3_parallel_2_threads_w_resnet_18_results/results/sacred/mappo/entombed_cooperative_v3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/entombed_cooperative_v3/eoi_entombed_cooperative_v3_w_episode_w_resnet18_results/results/sacred/eoi/entombed_cooperative_v3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/entombed_cooperative_v3/qmix_entombed_cooperative_v3_w_max_cycles=2500_w_resnet18_buffer_size=2000_results/sacred/qmix/entombed_cooperative_v3",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/entombed_cooperative_v3/entombed_cooperative_v3_w_max_cycles=2500_w_resnet18_buffer_size=1500_results/sacred/qplex/entombed_cooperative_v3"
    #                     ]
    # algo_names_ = ["coma", "maa2c", "mappo", "eoi", "qmix", "qplex"]
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
    # algo_names_ = ["maa2c", "mappo", "coma", "qmix", "eoi", "qplex"]
    # env_name_ = "space_invaders_v2"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/space_invaders_v2/"

    # Cramped_room
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/cramped_room/cramped_room_w_parallel_2_threads_results/results/sacred/maa2c/cramped_room",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/cramped_room/mappo_cramped_room_w_parallel_2_threads_results/sacred/mappo/cramped_room",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/cramped_room/coma_cramped_room_w_parallel_2_threads_results/sacred/coma/cramped_room",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/cramped_room/qmix_cramped_room_results/sacred/qmix/cramped_room",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/cramped_room/cramped_room_results/sacred/qplex/cramped_room",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/cramped_room/cramped_room_w_episode_results/sacred/eoi/cramped_room"
    # ]
    # algo_names_ = ["maa2c", "mappo", "coma", "qmix", "qplex", "eoi"]
    # env_name_ = "cramped_room"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/cramped_room/"

    # Assymetric_advantages
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/assymetric_advantages/maa2c_assymetric_advantages_w_max_steps=100M_w_parallel_2_threads_results/results/sacred/maa2c/asymmetric_advantages",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/assymetric_advantages/mappo_assymetric_advantages_w_max_steps=100M_w_parallel_2_threads_results/sacred/mappo/asymmetric_advantages",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/assymetric_advantages/coma_assymetric_advantages_w_max_steps=100M_w_parallel_2_threads_results/sacred/coma/asymmetric_advantages",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/assymetric_advantages/qmix_assymetric_advantages_w_max_steps=100M_results/sacred/qmix/asymmetric_advantages",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/assymetric_advantages/qplex_assymetric_advantages_w_max_steps=100M_results/sacred/qplex/asymmetric_advantages"
    # ]
    # algo_names_ = ["maa2c", "mappo", "coma", "qmix", "qplex"]
    # env_name_ = "assymetric_advantages"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/assymetric_advantages/"

    # Coordination ring
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/coordination_ring/maa2c_coordination_ring_w_max_steps=100M_w_parallel_2_threads_results/sacred/maa2c/coordination_ring",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/coordination_ring/mappo_coordination_ring_w_max_steps=100M_w_parallel_2_threads_results/sacred/mappo/coordination_ring",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/coordination_ring/coma_coordination_ring_w_max_steps=100M_w_parallel_2_threads_results/sacred/coma/coordination_ring",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/coordination_ring/coordination_ring_w_max_steps=100M_results/sacred/qmix/coordination_ring",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/coordination_ring/coordination_ring_w_max_steps=100M_results/sacred/qplex/coordination_ring",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/EOI/coordination_ring/eoi_coordination_ring_w_max_steps=100M_w_episode_results/sacred/eoi/coordination_ring"
    # ]
    # algo_names_ = ["maa2c", "mappo", "coma", "qmix", "qplex", "eoi"]
    # env_name_ = "coordination_ring"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/coordination_ring/"

    # PressurePlate - linear-4p-v0
    paths_to_results_ = [
        "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/pressureplate_linear-4p-v0/qmix_linear-4p-v0_results/sacred/qmix/pressureplate-linear-4p-v0",
        "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/pressureplate_linear-4p-v0/maa2c_linear-4p-v0_w_parallel_2_threads_results/sacred/maa2c/pressureplate-linear-4p-v0",
        "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/pressureplate_linear-4p-v0/coma_linear-4p-v0_results/sacred/coma/pressureplate-linear-4p-v0",
        "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/pressureplate_linear-4p-v0/qplex_linear-4p-v0_results/sacred/qplex/pressureplate-linear-4p-v0",
        "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAPPO/pressureplate_linear-4p-v0/mappo_linear-4p-v0_w_parallel_2_threads_results/sacred/mappo/pressureplate-linear-4p-v0"
    ]
    algo_names_ = ["qmix", "maa2c", "coma", "qplex", "mappo"]
    env_name_ = "pressureplate-linear-4p-v0"
    path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/pressureplate-linear-4p-v0/"


    # PressurePlate - linear-6p-v0
    # paths_to_results_ = [
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/MAA2C/pressureplate_linear-6p-v0/maa2c_linear-6p-v0_w_parallel_2_threads_results/sacred/maa2c/pressureplate-linear-6p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QMIX/pressureplate_linear-6p-v0/qmix_linear-6p-v0_results/sacred/qmix/pressureplate-linear-6p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/COMA/pressureplate_linear-6p-v0/coma_linear-6p-v0_results/sacred/coma/pressureplate-linear-6p-v0",
    #     "/home/georgepap/PycharmProjects/epymarl_master/experiments/results/QPLEX/pressureplate_linear-6p-v0/qplex_linear-6p-v0_results/sacred/qplex/pressureplate-linear-6p-v0"
    # ]
    # algo_names_ = ["maa2c", "qmix", "coma", "qplex"]
    # env_name_ = "pressureplate-linear-6p-v0"
    # path_to_save_ = "/home/georgepap/PycharmProjects/epymarl_master/experiments/multiple-exps-plots/pressureplate-linear-6p-v0/"

    plot_train_ = False
    plot_multiple_experiment_results(paths_to_results_, algo_names_, env_name_, path_to_save_, plot_train_)

