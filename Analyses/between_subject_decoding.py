# %% imports
import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import os
import glob
import fnmatch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import nibabel as nib
from nilearn.image import clean_img
from nilearn.signal import clean
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.masking import compute_epi_mask


from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    PredefinedSplit,
    KFold,
    GroupKFold,
)
from sklearn.feature_selection import SelectFpr, f_classif, SelectKBest
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    auc,
    roc_curve,
)

# %% constants
subIDs = ["004", "005", "006", "024", "026"]
exp_phases = ["rest", "preremoval", "study", "postremoval"]
stim_labels = {0: "Rest", 1: "Scene", 2: "Face"}
op_labels = {0: "Rest", 1: "Maintain", 2: "Replace", 3: "Suppress"}
stim_onscreen = {0: "Rest", 1: "Image", 2: "Operation", 3: "ITI"}
brain_space = {"T1w": "T1w", "MNI": "MNI152NLin2009cAsym"}
descs = ["preproc_bold", "brain_mask"]
ROIs = ["wholebrain"]
# n_runs = np.arange(3) + 1
n_runs = 3
tr_shift = [5]
shift_TR = tr_shift[0]
TR = [1]
op_TR = TR[0]
TR_run = [366]
op_TRs_run = TR_run[0]

# %% set up paths
data_path = "/Users/cnj678/Desktop/repclear_caleb/"
sub_design_path = "/Users/cnj678/Desktop/repclear_caleb/subject_designs/"
results_path = "/Users/cnj678/Desktop/repclear_caleb/caleb_replicate/results/"
combinedMask_path = "/Users/cnj678/Desktop/repclear_caleb/combined_masks/"


# %% create a function to find a file of a given path
def find_file(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
        return result


# %% create a function to simultaneously load label df and time shift
def get_shifted_labels(task, shift_size_TR, rest_tag=0):
    # shift labels
    def shift_timing(label_df, TR_shift_size, tag=0):
        nvars = len(label_df.loc[0])
        shift = pd.DataFrame(
            np.zeros((TR_shift_size, nvars)) + tag, columns=label_df.columns
        )
        shifted = pd.concat([shift, label_df])
        # remove timepoints outside of scanner time after shift
        return shifted[: len(label_df)]

    print("Loading labels...")

    # assign which df from the task to pull
    if task == "preremoval":
        task_tag = "pre-localizer"
    if task == "postremoval":
        task_tag = "post-localizer"
    if task == "study":
        task_tag = "study"

    sub_design = f"*{task_tag}*events*"
    sub_design_file = find_file(sub_design, sub_design_path)
    sub_design_matrix = pd.read_csv(sub_design_file[0])

    shifted_df = shift_timing(sub_design_matrix, shift_size_TR, rest_tag)

    print("Labels have been loaded")
    print(sub_design)
    print(sub_design_file)
    print(sub_design_matrix)

    return shifted_df


# %% load in bold file for a given run
def load_epi_data(subID, run):
    epi_in = os.path.join(
        data_path,
        f"sub-{subID}",
        "func",
        f"sub-{subID}_task-study_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    )
    epi_data = nib.load(epi_in)
    print("Loading data from %s" % (epi_in))
    return epi_data


# %% load mask data for a given roi
def load_mask(ROI, subID):
    for run in range(1, n_runs + 1):
        assert ROI in ROIs
        maskfile = os.path.join(
            combinedMask_path,
            "AllSubs_task-study_space-MNI152NLin2009cAsym_desc_brain_mask_COMBINED.nii.gz",
        )
        mask = nib.load(maskfile)
        print("Loaded %s mask for run %s" % (ROI, run))
    return mask


# %% create a function to mask the input data with the input mask
def mask_data(epi_data, mask):
    nifti_masker = NiftiMasker(mask_img=mask)
    epi_masked_data = nifti_masker.fit_transform(epi_data)
    return epi_masked_data


# %% create a function to load bold data, apply mask, and z-score data
def load_data(directory, subject_name, mask_name="", num_runs=3, zscore_data=False):
    # cycle through the masks
    print("Processing Start ...")

    # if there is a mask supplied, load it now
    if mask_name is "":
        mask = None
    else:
        mask = load_mask(mask_name, subject_name)

    # cycle through the runs
    for run in range(1, num_runs + 1):
        epi_data = load_epi_data(subject_name, run)

        # mask the data if necessary
        if mask_name is not "":
            epi_mask_data = mask_data(epi_data, mask)
        else:
            # do a whole brain mask
            if run == 1:
                # compute mask from epi
                mask = compute_epi_mask(epi_data).get_data()
            else:
                # get the intersection mask
                # (set voxels that are within the mask on all runs to 1, set all other voxels to 0)
                mask *= compute_epi_mask(epi_data).get_data()

            # reshape all of the data from 4D (X*Y*Z*time) to 2D (voxel*time): not great for memory
            epi_mask_data = epi_data.get_data().reshape(
                mask.shape[0] * mask.shape[1] * mask.shape[2], epi_data.shape[3]
            )

        # transpose and z-score (standardize) the data
        if zscore_data == True:
            scaler = preprocessing.StandardScaler().fit(epi_mask_data)
            preprocessed_data = scaler.transform(epi_mask_data)
        else:
            preprocessed_data = epi_mask_data

        # concatenate the data
        if run == 1:
            concatenated_data = preprocessed_data
        else:
            concatenated_data = np.vstack((concatenated_data, preprocessed_data))

    # apply the whole-brain masking: First, reshape the mask from 3D (X*Y*Z) to 1D (voxel)
    # second, get indices of non-zero voxels, i.e., voxels inside the mask
    # third, zero out all the voxels outside the mask
    if mask_name is "":
        mask_vector = np.nonzero(
            mask.reshape(
                mask.shape[0] * mask.shape[1] * mask.shape[2],
            )
        )[0]
        concatenated_data = concatenated_data[mask_vector, :]

    print("Data has been loaded for %s runs" % (num_runs))

    # save out to data folder
    np.save(
        f"/Users/cnj678/Documents/GitHub/SDS384_FinalProject_Team2/Analyses/data/operation decoding/bold/sub{subject_name}_study_MNI_masked_bold.npy",
        concatenated_data,
    )

    # return the list of mask data
    return concatenated_data, mask

# %%
for i in range(len(subIDs)):
    sub = subIDs[i]
    subID = sub

    bold_mask_data_all, _ = load_data(
        directory=data_path,
        subject_name=subID,
        mask_name="wholebrain",
        zscore_data=True
    )