import numpy as np
import matplotlib.pyplot as plt

def fig_wedge_photometry(wedge_counts, wedge_errs):
    fig, ax = plt.subplots(1,1, figsize=(8,8), subplot_kw={"polar": True})
    assert len(wedge_counts) == len(wedge_errs)
    nwedges = len(wedge_counts)
    theta = np.linspace(0, 2 * np.pi, nwedges, endpoint=False)
    width = 2 * np.pi / nwedges
    ax.set_theta_zero_location('N')
    ax.tick_params(labelsize=20,labelcolor='b',pad=12)
    barcolor = "orange"
    bars = ax.bar(
            theta,
            wedge_counts,
            width=width,
            # bottom=inner_radius,
            # yerr=wedgeErr,
            color=barcolor,
            alpha=1.0,
            edgecolor='black',
            align='edge',
            label="Counts"
        )
    for bar, std in zip(bars, wedge_errs):
        # Plot the 3-sigma errorbars for each wedge region
        bar_center = bar.get_x() + bar.get_width() / 2
        bar_height = bar.get_height()
        ax.errorbar(
            bar_center, bar_height, yerr=std*3, fmt='--', color="black", ecolor="black", capsize=3
        )

    plt.tight_layout()
    # plt.show()
    