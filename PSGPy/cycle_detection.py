#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for cycle detection and integration with hypnograms

@author: Pranay S. Yadav
"""
import pandas as pd
import numpy as np

# %% Cycle detection
def _detect_NREM_runs(hypno, min_length):
    """
    Identify successive runs of NREM (N1, N2 and N3) of minimum specified duration.

    Stage 1 of detecting NREM cycles - minimum continuous duration

    Parameters
    ----------
    hypno : pd.DataFrame
        Hypnogram dataframe obtained through load_hypno().
    min_length : int or float, positive non-zero
        Minimum duration in minutes for thresholding lengths of NREM runs.

    Returns
    -------
    runs : pd.DataFrame
        Successive NREM runs with onsets and offsets, thresholded based on durations.

    """
    # Specify filtering conditions
    condition = (hypno["Sleep"] == "NREM") & (hypno["Sleep_Runlength"] > min_length)

    # Group by runs based on condition and pick the first stage of each group
    runs = hypno.loc[condition].groupby("Sleep_Run").apply(lambda x: x.iloc[0])

    # Add offset by adding onset and run length
    runs["Offset"] = runs["Onset"] + runs["Sleep_Runlength"]

    return runs


def _segregate_NREM_runs(runs, min_separation):
    """
    Differentiate successive runs of NREM based on gap durations separating them.

    Stage 2 of detecting NREM cycles - minimum separation of successive runs

    Parameters
    ----------
    runs : pd.DataFrame
        Successive NREM runs with onsets and offsets, thresholded based on durations.
        Output of _detect_NREM_runs()
    min_separation : int or float, positive non-zero
        Minimum duration in minutes for thresholding gaps between consecutive NREM runs.

    Returns
    -------
    runs : pd.DataFrame
        Updated with NREM cycle numbers and gap from next run.

    """
    # Calculate gap or separation between successive runs
    runs["Next_Run"] = runs["Onset"].shift(-1) - runs["Offset"]

    # Specify filtering condition
    condition = runs["Next_Run"] > min_separation

    # Get numbered cycles based on condtion, starting from 1
    cycle_numbers = (1 + (condition).cumsum()).shift(fill_value=1)

    # Add a string identifier column
    runs["CYC"] = "CYC_" + (cycle_numbers).astype("str")

    return runs


def _detect_cycle_offsets(hypno, runs):
    """
    Detect offsets for cycles based on REM or awakenings.

    For each cycle, onset corresponds to onset of NREM cycle, offset depends on three
    sequential criteria:
        1. Offset of REM cycle
        2. Offset of last N3 prior to Short Awakening
        3. Onset of Long Awakening

    Parameters
    ----------
    hypno : pd.DataFrame
        Hypnogram dataframe obtained through load_hypno().
    runs : pd.DataFrame
        NREM runs processed for minimum duration & separation using _detect_NREM_runs()
        and _segregate_NREM_runs() respectively.

    Returns
    -------
    cycles : list of dict
        Estimates for onset, offset, duration of each detected cycle.

    """
    # Add an upper limit for the last cycle as end of recording
    runs.loc[runs.index[-1], "Next_Run"] = (
        hypno.iloc[-1].loc[["Onset", "Duration"]].sum()  # End of recording
        - runs.loc[runs.index[-1], "Offset"]  # End of last NREM cycle
    )

    # Initialize aggregator
    cycles = []

    # Loop over groupby object
    for idx, chunk in runs.groupby(["CYC"]):

        # Onset of cycle is the first row -> onset of 1st sufficiently long NREM stage
        lowerlim = chunk.iloc[0]["Onset"]

        # Upper limit for offset detection is the onset of next cycle which corresponds
        # to the offset of the last run in cycle plus distance to next run
        upperlim = chunk.iloc[-1][["Offset", "Next_Run"]].sum()

        # Extract subset of data based on limits to work with
        dat_win = hypno.loc[(hypno["Onset"] > lowerlim) & (hypno["Onset"] < upperlim)]

        # Condition 1: REM offset
        if any(dat_win["Sleep"] == "REM"):

            # Get last REM
            cycle_offset = dat_win.loc[dat_win["Sleep"] == "REM", :].iloc[-1, :]

            # Get offset in minutes, indices and hypnogram entries
            offset_min = cycle_offset.loc[["Onset", "Duration"]].sum()
            offset = "Last_REM"
            offset_entry = cycle_offset["Entry"]
            offset_idx = cycle_offset.name

            # Switch to indicate offset found
            switch = True

        # Condition 2: N3 followed by short awakening
        elif any(dat_win["Stage"] == "N3") and any(  # Check if N3-W transition present
            dat_win["Stage"] + "_" + dat_win["Stage"].shift(-1) == "N3_W"
        ):

            # Shift sequence and find the first N3-W transition
            condition = dat_win["Stage"] + "_" + dat_win["Stage"].shift(-1) == "N3_W"
            last_N3_onset = dat_win.iloc[np.where(condition)[0][0]]

            # End of first cycle is end of last N3 before awakening
            offset_min = last_N3_onset[["Onset", "Duration"]].sum()
            offset = "Last_N3_Before_Awakening"
            offset_entry = last_N3_onset["Entry"]
            offset_idx = last_N3_onset.name

            # Switch to indicate offset found
            switch = True

        # Condition 3: Long awakening
        elif any(dat_win["Awakening"] == "Long"):

            # Get first long awakening
            cycle_offset = dat_win.loc[dat_win["Awakening"] == "Long", :].iloc[0, :]

            # Get offset in minutes, indices and hypnogram entries
            offset_min = cycle_offset.loc["Onset"]
            offset = "Long_Awakening"
            offset_entry = cycle_offset["Entry"] - 1
            offset_idx = cycle_offset.name - 1

            # Switch to indicate offset found
            switch = True

        # Else cycle doesn't end
        else:
            switch = False

        # If offset found, then prepare dictionary for aggregation
        if switch:
            Cx = {
                "Cycle": idx,
                "Onset": lowerlim,
                "Offset": offset_min,
                "Duration": offset_min - lowerlim,
                "Offset_Mode": offset,
                "Onset_Entry": chunk.iloc[0]["Entry"],
                "Offset_Entry": offset_entry,
                "Onset_idx": chunk.iloc[0]["Entry"] - 1,
                "Offset_idx": offset_idx,
            }

            # Aggregate
            cycles.append(Cx)

    return cycles


def detect_cycles(hypno, min_length=10, min_separation=10):
    """
    Detect onsets and offsets for cycles based on thresholds & deterministic criteria.

    For each cycle, onset corresponds to onset of NREM cycle, offset depends on three
    sequential criteria:
        1. Offset of REM cycle
        2. Offset of last N3 prior to Short Awakening
        3. Onset of Long Awakening

    Parameters
    ----------
    hypno : pd.DataFrame
        Hypnogram dataframe obtained through load_hypno().
    min_length : int or float, positive non-zero, optional
        Minimum duration in minutes for thresholding lengths of NREM runs.
        The default is 10.
    min_separation : int or float, positive non-zero, optional
        Minimum duration in minutes for thresholding gaps between consecutive NREM runs.
        The default is 10.

    Returns
    -------
    pd.DataFrame
        Tabulated estimates for onsets, offsets and durations of detected cycles.

    """
    # Detect long-enough consecutive NREM runs
    runs = _detect_NREM_runs(hypno, min_length)

    # Detect NREM cycles based on long-enough gaps between successive NREM runs
    runs = _segregate_NREM_runs(runs, min_separation)

    # Detect cycles offsets
    cycles = _detect_cycle_offsets(hypno, runs)

    return pd.DataFrame(cycles)


# %% Merge cycle information with hypnogram
def update_hypnogram_cycles(hypno, cycles):
    """
    Add cycle identifiers to hypnogram based on cycle detections

    Parameters
    ----------
    hypno : pd.DataFrame
        Hypnogram dataframe obtained through load_hypno().
    cycles : pd.DataFrame
        Tabulated estimates for onsets, offsets and durations of detected cycles.

    Returns
    -------
    hypno : pd.DataFrame
        Hypnogram dataframe with updated cycle metadata

    """
    # Iterate over each cycle
    for k, cycle in cycles.iterrows():

        # Add onset, offset, mode of offset and durations to hypnogram
        start = cycle["Onset_idx"]
        end = cycle["Offset_idx"]
        hypno.loc[start:end, "Cycle_Num"] = cycle["Cycle"]
        hypno.loc[start:end, "Cycle_Duration"] = cycle["Duration"]
        hypno.loc[start:end, "Cycle_Offset_Mode"] = cycle["Offset_Mode"]

    return hypno
