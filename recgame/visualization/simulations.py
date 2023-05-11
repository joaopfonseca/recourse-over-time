import numpy as np
from rich.progress import track
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def simulations_scores_panel(results, adaptation_list, new_agents_list):
    """
    TODO:
    - Documentation
    - Remove adaptation_list and new_agents_list from parameters (they are not necessary)
    """
    fig, axes = plt.subplots(
        len(adaptation_list),
        len(new_agents_list),
        figsize=(10, 10),
        layout="constrained",
        sharey=True,
        sharex=True,
    )

    for params, env in track(results):
        i = adaptation_list.index(params["adaptation"])
        j = new_agents_list.index(params["new_agents"])

        ax = env.plot.agent_scores(title=False, legend=False, ax=axes[i, j])
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticks(range(0, 51, 10))
        ax.set_xlim(-1, 51)

    for i in range(len(adaptation_list)):
        axes[i, 0].set_ylabel(f"{adaptation_list[i]}")

    for j in range(len(new_agents_list)):
        axes[-1, j].set_xlabel(f"{new_agents_list[j]}")

    fig.supxlabel(r"Number of New Agents")
    fig.supylabel("Adaptation")

    return fig, axes


def simulations_success_rate_panels(results, adaptation_list, new_agents_list):
    """
    TODO:
    - Documentation
    - Remove adaptation_list and new_agents_list from parameters (they are not necessary)
    """
    fig, axes = plt.subplots(
        len(adaptation_list),
        len(new_agents_list),
        figsize=(10, 10),
        layout="constrained",
        sharey=True,
        sharex=True,
    )

    for params, env in track(results):
        i = adaptation_list.index(params["adaptation"])
        j = new_agents_list.index(params["new_agents"])
        sr = env.success_rate(1, env.step_ + 1)
        sr_smooth = gaussian_filter1d(
            np.where(np.isnan(sr), sr[~np.isnan(sr)].mean(), sr), sigma=5
        )

        ax = axes[i, j]
        ax.plot(range(1, env.step_ + 1), sr, alpha=0.25)
        ax.plot(range(1, env.step_ + 1), sr_smooth, c="#d75c5c")
        ax.set_xticks(range(0, 51, 10))
        ax.set_xlim(-1, 51)

    for i in range(len(adaptation_list)):
        axes[i, 0].set_ylabel(f"{adaptation_list[i]}")

    for j in range(len(new_agents_list)):
        axes[-1, j].set_xlabel(f"{new_agents_list[j]}")

    # fig.suptitle("Success rate")
    fig.supxlabel(r"Number of New Agents")
    fig.supylabel("Adaptation")

    return fig, axes
