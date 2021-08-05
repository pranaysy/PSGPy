#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for reading and processing hypnograms

@author: Pranay S. Yadav
"""
import pandas as pd
import numpy as np
from mne import read_annotations

# %% Read and process raw hypnogram stored as EDF
def read_raw_hypnogram(edf_file):
    """
    This function read hypnograms in EDF format exported from Polyman using mne-python's
    read_annotations interface, returns a pandas dataframe object with hypnogram data

    Parameters
    ----------
    edf_file : pathlib.Path object
        Location to EDF file containing hypnogram.

    Returns
    -------
    pd.DataFrame
        Tabulated hypnogram with 4 columns:
            0. Entry index of observed sleep stage
            1. Onset in minutes
            2. Duration in minutes
            3. Stage in alphanumeric characters

    """

    # Read EDF annotation using MNE's builtin, as a dataframe
    df_hypno = pd.DataFrame(read_annotations(str(edf_file)))

    # Check if dataframe is empty, if MNE doesn't find valid annotations
    # it will return an empty annotation object
    if df_hypno.empty:
        print(f"ERROR: Invalid or empty hypnogram data for\n  {edf_file}")
        return df_hypno

    # Drop unwanted column
    df_hypno.drop(columns="orig_time", inplace=True)

    # Set Onset_hr & Duration_min to the correct units (hours)
    df_hypno.onset = df_hypno.onset / 60
    df_hypno.duration = df_hypno.duration / 60

    # Trim the stage column to single words
    df_hypno.description = df_hypno.description.str.split(" ", expand=True)[2]

    # Rename columns
    df_hypno.rename(
        columns={"onset": "Onset", "duration": "Duration", "description": "Stage"},
        inplace=True,
    )

    # Index from 1 & assign identity
    df_hypno.index = df_hypno.index + 1
    df_hypno.index.name = "Entry"

    # Handle the case where NREM stages are labelled as integers
    alt_map = {"1": "N1", "2": "N2", "3": "N3"}
    df_hypno["Stage"] = df_hypno["Stage"].replace(alt_map)

    # Remap stages to integers and store separately
    hypno_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
    df_hypno["StageN"] = df_hypno["Stage"].replace(hypno_map)

    # Shift 'Entry' from index to column and return
    return df_hypno.reset_index()


# %% Read and prep hypnogram
def _read_hypno(file):
    """
    Load a hypnogram file stored as raw EDF or as tabulated csv

    Parameters
    ----------
    file : pathlib.Path object
        Location of EDF or csv file containing hypnogram.

    Returns
    -------
    hypno : pd.DataFrame
        Processed hynpogram.

    """
    # Switch based on file format
    if file.suffix.lower() == ".csv":

        # Read CSV
        hypno = pd.read_csv(file)

        # Sanity check for column headers
        anticipated_columns = {"Entry", "Onset", "Duration", "Stage", "StageN"}
        if not set(hypno.columns).issubset(anticipated_columns):
            raise TypeError("CSV does not contain relevant hypnogram information!")

    elif file.suffix.lower() == ".edf":

        # Read EDF
        hypno = read_raw_hypnogram(file)

    else:
        raise ValueError("File type not recognized - only CSV or EDF are valid!")

    # Assign Sleep types for easier grouping
    mapping = {"W": "Wake", "N1": "NREM", "N2": "NREM", "N3": "NREM", "R": "REM"}
    hypno["Sleep"] = hypno["Stage"].replace(mapping)

    return hypno


def _identify_runs(hypno):
    """
    Identify runs of successive sleep types, number them and add lengths of each run.
    Grouping of runs done for each of Wake, NREM and REM.

    Parameters
    ----------
    hypno : pd.DataFrame
        Hypnogram dataframe obtained through _read_hypno().

    Returns
    -------
    hypno : pd.DataFrame
        Hypnogram with added columns for successive sleep runs and lengths.

    """
    # Get successively non-identifical sleep types -> Add up to get runs/chains
    hypno.loc[:, "Run"] = hypno["Sleep"].ne(hypno["Sleep"].shift()).cumsum()

    # Merge run number with sleep type for unique identification
    hypno["Sleep_Run"] = hypno["Sleep"] + "_" + hypno["Run"].astype("str").str.zfill(3)

    # Add length of each run using groupby and transform with sum
    hypno["Sleep_Runlength"] = hypno.groupby("Sleep_Run")["Duration"].transform("sum")

    return hypno


def _flag_awakenings(hypno, thresh):
    """
    Mark awakenings as Long or Short depending on threshold in minutes

    Parameters
    ----------
    hypno : pd.DataFrame
        Hypnogram dataframe obtained through _read_hypno().
    thresh : int or float, positive non-zero
        Minimum duration in minutes for awakenings to qualify as long.

    Returns
    -------
    hypno : pd.DataFrame
        Hypnogram with added columns for type of awakening.

    """
    # Sanity checks
    assert isinstance(thresh, (int, float)), "Threshold should be a numeric type"
    assert thresh > 0, "Threshold should be a positive number (minutes)"

    # Seggregate long and short awakenings based on a 2 minute threshold
    wake_mask = hypno["Stage"] == "W"
    hypno.loc[(wake_mask) & (hypno["Duration"] <= thresh), "Awakening"] = "Short"
    hypno.loc[(wake_mask) & (hypno["Duration"] > thresh), "Awakening"] = "Long"

    return hypno


def load_hypnogram(file, wake_thresh=2):
    """
    Load a hypnogram stored as tabulated CSV and prepare it for downstream processing.

    Wrapper around IO and two functions for identifying runs and awakenings.

    Parameters
    ----------
    file : pathlib.Path object
        Location of csv file containing hypnogram.
    wake_thresh : int or float, positive non-zero, optional
        Minimum duration in minutes for awakenings to qualify as long. The default is 2.

    Returns
    -------
    hypno : pd.DataFrame
        Hypnogram with added columns for:
            - type of awakening
            - successive sleep runs
            - successive sleep run lengths

    """
    # Read hypnogram from csv file
    hypno = _read_hypno(file)

    # Identify successive runs
    hypno = _identify_runs(hypno)

    # Mark awakenings as long or short
    hypno = _flag_awakenings(hypno, thresh=wake_thresh)

    return hypno


# %% Resample hypnogram epoch-wise
def resample_hypnogram(hypno, cycles=None):
    """
    Process hypnogram and return a "sampled" array of sleep stages with numeric mapping

    Parameters
    ----------
    hypno : pd.DataFrame
        Hypnogram data read and processed by the function read_hypnogram.
    cycles : pd.DataFrame, optional
        Tabulated estimates for onsets, offsets and durations of detected cycles.
        The default is None

    Returns
    -------
    pd.DataFrame
        Hypnogram resampled to epoch time units and cycle information, if provided

    """
    unit = 2  # 1 minute = 2 epochs for resampling

    # Get times
    ntimes = (hypno["Duration"] * unit).astype("int")

    # Get total duration of hypnogram in samples
    total_durn = (ntimes).sum()
    total_durn_alt = (hypno[["Onset", "Duration"]].iloc[-1].sum() * unit).astype("int")
    assert total_durn == total_durn_alt, "Inconsistent durations"

    # Convert duration-wise stage labels to "sampled" labels (stage id for each sample)
    if cycles is not None:
        resampled = [
            np.repeat(hypno["Stage"], repeats=ntimes),
            np.repeat(hypno["StageN"], repeats=ntimes),
            np.repeat(hypno["Cycle_Num"], repeats=ntimes),
        ]
    else:
        resampled = [
            np.repeat(hypno["Stage"], repeats=ntimes),
            np.repeat(hypno["StageN"], repeats=ntimes),
        ]

    # Transform resampled arrays to tidy dataframe
    hypno_resampled = (
        pd.concat(resampled, axis=1,)
        .reset_index()
        .rename(columns={"index": "Hypnogram_Entry"})
        .reset_index()
        .rename(columns={"index": "Epoch_number"})
    )

    return hypno_resampled
