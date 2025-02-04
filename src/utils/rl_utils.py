import torch as th
import numpy as np


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]


def compute_stats(data, min_val, max_val):
    means = [m for m, s in data if m is not None]
    normalized_means = [(m - min_val) / (max_val - min_val) if max_val > min_val else 0 for m in means]

    if len(means) == 0:
        return np.nan, np.nan

    mean_avg = np.mean(normalized_means)
    std = 1.15 * np.std(normalized_means) / np.sqrt(len(normalized_means))

    return mean_avg, std

def print_latex_table(all_algorithms, environments):
    print(
        "\\textbf{Algorithms\\textbackslash Environments} & \\textbf{LBF} & \\textbf{RWARE} & \\textbf{Spread (MPE)} & \\textbf{Overcooked} & \\textbf{Petting Zoo} & \\textbf{Pressure Plate}\\\\")
    print("\\midrule")

    for alg in all_algorithms:
        print(f"\\textbf{{{alg}}}", end="")
        for env_name in environments.keys():
            algo_data = environments[env_name].get(alg)
            if algo_data:
                all_means = [mean for data in environments[env_name].values() for mean, _ in data if mean is not None]
                min_val = min(all_means) if all_means else np.nan
                max_val = max(all_means) if all_means else np.nan

                avg, adj_std = compute_stats(algo_data, min_val, max_val)
                if np.isnan(avg) or np.isnan(adj_std):
                    print(" & N/A", end="")
                else:
                    print(f" & \({avg:.2f} \pm {adj_std:.2f}\)", end="")
            else:
                print(" & N/A", end="")
        print(" \\\\")

if __name__ == '__main__':

    environments = {
        "LBF": {
            "QMIX": [(0.94, 0.09), (0.00, 0.00), (0.89, 0.01), (0.08, 0.19), (0.01, 0.00), (0.02, 0.01), (0.06, 0.05)],
            "QPLEX": [(0.63, 0.48), (0.60, 0.49), (0.98, 0.00), (0.00, 0.00), (0.00, 0.00), (0.01, 0.00), (0.00, 0.00)],
            "MAA2C": [(0.98, 0.02), (0.59, 0.38), (0.81, 0.03), (0.00, 0.00), (0.78, 0.02), (0.52, 0.24), (0.71, 0.02)],
            "MAPPO": [(0.64, 0.34), (0.18, 0.35), (0.77, 0.04), (0.00, 0.00), (0.57, 0.18), (0.41, 0.23), (0.57, 0.03)],
            "HAPPO": [(None, None), (0.00, 0.00), (0.71, 0.02), (0.00, 0.00), (0.00, 0.00), (None, None), (0.00, 0.00)],
            "MAT-DEC": [(0.28, 0.14), (0.54, 0.02), (0.55, 0.06), (0.00, 0.00), (0.29, 0.09), (0.04, 0.04), (0.08, 0.06)],
            "COMA": [(0.00, 0.00), (0.00, 0.00), (0.03, 0.02), (0.00, 0.00), (0.03, 0.01), (0.03, 0.00), (0.02, 0.00)],
            "EOI": [(0.00, 0.00), (0.00, 0.00), (0.34, 0.23), (0.00, 0.00), (0.03, 0.01), (0.07, 0.00), (0.04, 0.00)],
            "MASER": [(0.00, 0.00), (0.00, 0.00), (0.01, 0.01), (0.00, 0.00), (0.01, 0.00), (0.01, 0.00), (0.01, 0.00)],
            "EMC": [(0.00, 0.00), (None, None), (0.63, 0.36), (None, None), (0.01, 0.00), (None, None), (0.00, 0.00)],
            "CDS": [(0.00, 0.00), (0.00, 0.00), (0.91, 0.02), (0.00, 0.00), (0.00, 0.00), (0.00, 0.00), (0.00, 0.00)]
        },
        "RWARE": {
            "QMIX": [(0.00, 0.00), (0.00, 0.00), (0.01, 0.01)],
            "QPLEX": [(0.61, 0.45), (11.73, 10.81), (0.94, 0.63)],
            "MAA2C": [(2.25, 0.62), (6.87, 7.12), (1.38, 1.26)],
            "MAPPO": [(2.91, 0.82), (17.1, 5.31), (4.14, 2.12)],
            "HAPPO": [(1.46, 2.06), (24.07, 0.71), (9.69, 3.19)],
            "MAT-DEC": [(6.05, 8.56), (27.85, 19.71), (0.00, 0.00)],
            "COMA": [(0.02, 0.00), (0.02, 0.00), (0.03, 0.00)],
            "EOI": [(6.58, 3.92), (12.09, 4.59), (0.17, 0.01)],
            "EMC": [(None, None), (None, None),(None, None)],
            "MASER": [(0.00, 0.00), (0.00, 0.00), (0.02, 0.00)],
            "CDS": [(4.00, 2.30), (27.37, 6.88), (4.11, 0.02)]
        },
        "Spread (MPE)": {
            "QMIX": [(-1278.26, 23.13), (-2531.17, 586.56), (-6414.48, 27.27)],
            "QPLEX": [(-766.84, 14.39), (-1800.53, 194.49), (-13260.36, 6200.03)],
            "MAA2C": [(-1190.09, 99.93), (-2312.56, 222.08), (-5961.67, 66.04)],
            "MAPPO": [(-971.17, 124.22), (-1910.20, 42.86), (-5926.39, 38.48)],
            "HAPPO": [(-1032.80, 45.84), (-2000.41, 98.20), (-6940.61, 69.55)],
            "MAT-DEC": [(-1066.62, 45.98), (-1918.88, 15.76), (-6843.44, 563.39)],
            "COMA": [(-1176.78, 33.37), (-2003.47, 51.18), (-6249.07, 44.73)],
            "EOI": [(-1963.23, 859.27), (None, None), (None, None)],
            "MASER": [(-969.06, 5.01), (None, None), (None, None)],
            "EMC": [(None, None), (None, None), (None, None)],
            "CDS": [(-864.96, 0.00), (None, None), (None, None)]
        },
        "Petting Zoo": {
            "QMIX": [(991.59, 0.17), (199.57, 0.61), (8.00, 0.00)],
            "QPLEX": [(991.46, 0.10), (197.78, 3.14), (8.00, 0.00)],
            "MAA2C": [(990.83, 0.04), (-0.33, 3.47), (6.57, 0.05)],
            "MAPPO": [(990.71, 0.10), (13.22, 4.61), (6.61, 0.05)],
            "HAPPO": [(983.60, 0.77), (25.62, 3.36), (None, None)],
            "MAT-DEC": [(982.57, 2.44), (200.00, 0.00), (None, None)],
            "COMA": [(678.28, 324.06), (1.10, 7.51), (7.68, 1.70)],
            "EOI": [(948.35, 42.74), (-1.64, 2.66), (6.53, 0.02)],
            "MASER": [(989.39, 0.64), (104.25, 69.29), (None, None)],
            "EMC": [(265.00, 174.58), (196.5, 2.83), (None, None)],
            "CDS": [(415.82, 284.37), (197.13, 2.26), (None, None)]
        },
        "Overcooked" : {
            "QMIX": [(0.00, 0.00), (300.00, 300.00), (0.00, 0.00)],
            "QPLEX": [(86.67, 122.57), (0.00, 0.00), (0.00, 0.00)],
            "MAA2C": [(286.80, 9.34), (487.80, 107.60), (0.10, 0.10)],
            "MAPPO": [(280.00, 0.00), (0.30, 0.10), (0.07, 0.09)],
            "HAPPO": [(0.00, 0.00), (None, None), (0.00, 0.00)],
            "MAT-DEC": [(0.00, 0.00), (0.00, 0.00), (0.00, 0.00)],
            "COMA": [(0.20, 0.16), (0.10, 0.10), (0.07, 0.09)],
            "EOI": [(280.0, 0.00), (1.60, 0.60), (0.13, 0.09)],
            "EMC": [(None, None), (None, None), (None, None)],
            "MASER": [(0.00, 0.00), (None, None), (0.00, 0.00)],
            "CDS": [(186.67, 133.00), (70.00, 70.00), (0.00, 0.00)]
        },
        "PressurePlate": {
            "QMIX": [(-210.72, 17.84), (-3461.77, 1020.33)],
            "QPLEX": [(-652.19, 9.29), (-5183.56, 345.46)],
            "MAA2C": [(-281.59, 201.77), (-547.39, 21.00)],
            "MAPPO": [(-135.99, 1.32), (-494.08, 10.54)],
            "HAPPO": [(None, None), (-584.90, 201.68)],
            "MAT-DEC": [(-876.58, 1113.53), (-2930.77, 3652.53)],
            "COMA": [(-4391.79, 108.96), (-12360.20, 314.06)],
            "EOI": [(-3050.64, 1125.83), (-9221.16, 4796.08)],
            "MASER": [(-88.44, 2.84), (-5257.08, 4099.19)],
            "EMC": [(-4518.23, 249.68), (-12347.60, 0.0)],
            "CDS": [(-1926.45, 260.85), (-8068.23, 519.79)]
        }
    }

    all_algorithms = set()
    for env in environments.values():
        all_algorithms.update(env.keys())

    print_latex_table(all_algorithms, environments)

