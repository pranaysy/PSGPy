#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for visualization of hypnograms and writing plots to disk

@author: Pranay S. Yadav
"""
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from matplotlib import pyplot as plt

# %% Set up plotting preferences
sns.set(
    style="whitegrid",
    context="paper",
    font="Helvetica Neue LT Com",
    font_scale=1.4,
    rc={
        "grid.color": "0.9",
        "grid.linewidth": "0.5",
        "axes.edgecolor": "0.0",
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "font.weight": 400,
        "font.style": "normal",
        "font.stretch": "condensed",
        "font.variant": "normal",
        "font.size": 10,
        "text.hinting": "force_autohint",
        "figure.autolayout": True,
        "savefig.dpi": 150,
    },
)

# %% Plot hypnogram
def plot_hypnogram(resampled_hypno, cycles=None, label="", fig=None, ax=None):
    """
    Plot an aesthetically pleasing hypnogram with optional cycle markers.

    Parameters
    ----------
    resampled_hypno : pd.DataFrame
        Hypnogram resampled to epoch time units.
    cycles : cycles : pd.DataFrame, optional
        Tabulated estimates for onsets, offsets and durations of detected cycles.
        The default is None.
    label : str, optional
        Custom label to add in plot title, usually a file identifier. The default is "".

    Returns
    -------
    fig : Figure object
        Main figure on which hypnogram has been plotted.
    ax : Axis object
        Axis on figure used by hypnogram.

    """
    if (fig is None) and (ax is None):

        # Initialize figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(16, 4.5), tight_layout=True, sharex=True)

    # Ordering of stages and corresponding colors
    stages = ["N3", "N2", "N1", "W", "R"]
    colors = ["#002D72", "#005EB8", "#307FE2", "#30B700", "#BE3A34"]

    # Step 1: Plot colored markers for each stage
    # Iterate over each stage-color pair
    for stage, color in zip(stages, colors):

        # Filter subset of data
        dat = resampled_hypno.loc[resampled_hypno["Stage"] == stage]

        # Plot markers for each stage
        ax.plot(
            dat["Epoch_number"],
            dat["Stage"],
            linestyle="",
            marker="s",
            color=color,
            markersize=2.5,
            alpha=0.5,
            # mec='w',
            zorder=10,
        )

    # Step 2: Plot stage changes - classic hypnogram style
    # Plot all stages across all epochs as a line
    ax.plot(
        resampled_hypno["Epoch_number"],
        resampled_hypno["Stage"],
        color="k",
        alpha=0.4,
        zorder=1,
        linestyle="-",
        linewidth=0.75,
    )

    # If cycle information provided
    if cycles is not None:

        # Iterate over cycles
        for k, row in cycles.iterrows():

            # Plot background tint
            ax.axvspan(
                xmin=2 * row["Onset"],  # Convert minutes to epochs
                xmax=2 * row["Offset"],  # Convert minutes to epochs
                ymin=0.025,
                ymax=0.975,
                color="#98B6E4",
                alpha=0.16,
            )

            # Plot cycle onset and offset, and add text label for cycle number
            ax.axvline(2 * row["Onset"], color="#98A4AE", alpha=0.25, linestyle="-")
            ax.axvline(2 * row["Offset"], color="#98A4AE", alpha=0.25, linestyle="-")
            ax.text(2 * row["Onset"] + 5, "R", f"C$_{k+1}$", va="top", alpha=0.75)

        # Prepare secondary title with durations
        cst = cycles["Duration"].sum()
        tst = resampled_hypno["Epoch_number"].iloc[-1] / 2
        N = len(cycles)
        covg = 100 * cst / tst
        title = f"{N} Sleep Cycles - Coverage: {covg:.1f}% of TST ({cst} of {tst} min)"
        ax.set_title(title, loc="right")

    # Adjust x-axis limits for cleaner fit, and add grid markers, ticks every 100 epochs
    ax.set_xlim(-5, resampled_hypno["Epoch_number"].iloc[-1] + 5)
    ax.xaxis.set_major_locator(MultipleLocator(100))

    # Sane axis labels and primary title
    ax.set_ylabel("Sleep Stage")
    ax.set_xlabel("Epoch Number")
    ax.set_title(f"Hypnogram for PSG {label}", loc="left", fontweight="bold")

    return fig, ax


def save_hypnogram_plot(fig, label, folder, tiff=True, svg=False, jpg=False):
    """
    Save a hypnogram plot in TIFF, SVG and JPG formats

    Parameters
    ----------
    fig : Figure object
        Main figure on which hypnogram has been plotted.
    label : str
        Custom label to add in plot title, usually a file identifier.
    folder : pathlib.Path object
        Path to output directory for saving hypnogram plots.

    Returns
    -------
    None.

    """
    out = folder / f"{label}_Hypnogram"

    if tiff:
        fig.savefig(out.with_suffix(".tiff"), pil_kwargs={"compression": "tiff_lzw"})

    if svg:
        fig.savefig(out.with_suffix(".svg"))

    if jpg:
        fig.savefig(out.with_suffix(".jpg"))
