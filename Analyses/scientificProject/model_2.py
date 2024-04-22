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

# %% paths
data_path = '/Users/cnj678/Desktop/data_ML_final/raw bold/'
sub_design_path = '/Users/cnj678/Desktop/data_ML_final/subejct_designs/'
combinedMask_path = '/Users/cnj678/Desktop/data_ML_final/combined_masks/'
results_path = '/Users/cnj678/Documents/GitHub/SDS384_FinalProject_Team2/Analyses/scientificProject/Results/'

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
    existing_bold_data = f'/Users/cnj678/Desktop/data_ML_final/concatenated/operation decoding/bold/sub{subject_name}_study_MNI_masked_bold.npy'
    if os.path.exists(existing_bold_data):
      concatenated_data = np.load(existing_bold_data)
      print('Existing bold data loaded!')

      # if there is a mask supplied, load it now
      if mask_name is "":
        mask = None
      else:
        mask = load_mask(mask_name, subject_name)

      return concatenated_data, mask

    else:
      print('File not found :(')

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
          f"/Users/cnj678/Desktop/data_ML_final/concatenated/operation decoding/bold/sub{subject_name}_study_MNI_masked_bold.npy",
          concatenated_data,
      )

      # return the list of mask data
      return concatenated_data, mask

# %% create function to do things similar to clearmem (feature selection = first half of each run; training/test = second half of each run)
def betweenSub_operation_sample(full_data, labels_df):
    # operations IDs - 1: maintain, 2: replace, 3: suppress
    # stim on screen - 1: image, 2: operation, 3: ITI

    # this will restrict us to only suppress trials
    suppress_only = [3]
    labels_df = labels_df[labels_df['condition'].isin(suppress_only)]

    operation_list = labels_df["condition"]
    stim_onscreen = labels_df["stim_present"]
    run_list = labels_df["run"]

    # define operation period
    operation_index = np.where(np.logical_or(stim_onscreen == 2, stim_onscreen == 3))[0]
    rest_index = []
    runs = run_list.unique()[1:]

    operation_val = operation_list.values[operation_index]
    run_val = run_list.values[operation_index]

    # we want the first half of each run for feature selection
    bold_data = []
    operation_sample = operation_val

    sub_loop = 0
    for sub in subIDs:
        print(sub)
        bold = full_data[f"{sub}"][operation_index]

        if sub_loop == 0:
            bold_data = bold
            print(bold_data.shape)
            sub_loop = sub_loop + 1
        else:
            bold_data = np.concatenate((bold_data, bold))
            print(bold_data.shape)

    subject_sample = np.repeat(range(0, 5), 300)
    op_labels = np.tile(operation_sample, 5)

    # here I am manually setting the group (good vs. bad) based on memory analyses at the end of the script
    # 0: bad suppressor;  1: good suppresor
    sub_004_group = np.ones(300)
    sub_005_group = np.ones(300)
    sub_006_group = np.zeros(300)
    sub_024_group = np.ones(300)
    sub_026_group = np.zeros(300)

    group_labels = np.concatenate((sub_004_group, sub_005_group, sub_006_group, sub_024_group, sub_026_group))

    print(bold_data.shape)
    print(group_labels.shape)
    print(subject_sample.shape)

    print(op_labels)

    return bold_data, group_labels, subject_sample

# %% set up decoder
def betweenSub_decode_model(bold_data, group_labels, subject_sample):
    scores = []
    decision_scores = []
    test_labels = []
    pred_probs = []
    confusion_matrices = []
    #aucs = []
    evidences = []
    #roc_aucs = []

    ps = PredefinedSplit(subject_sample)
    for train, test in ps.split():
        train_data = bold_data[train]
        test_data = bold_data[test]
        train_label = group_labels[train]
        test_label = group_labels[test]
        print(train_data.shape)

        # feature selection
        #Fselect_fpr = SelectFpr(f_classif, alpha=0.001).fit(train_data, train_label)
        #bold_train_subject = Fselect_fpr.transform(train_data)
        #bold_test_subject = Fselect_fpr.transform(test_data)

        bold_train_subject = train_data
        bold_test_subject = test_data

        print(bold_train_subject.shape)
        print(bold_test_subject.shape)

        # train with selected penalty
        log_reg = LogisticRegression(penalty="l2", solver="lbfgs", C=50, max_iter=1000)
        log_reg.fit(bold_train_subject, train_label)

        # now test on the held out subject
        score = log_reg.score(bold_test_subject, test_label)
        decision_score = log_reg.decision_function(bold_test_subject)

        classes = np.unique(test_label).size

        # calculate auc for each operation
        #fpr = dict()
        #tpr = dict()
        #roc_auc = dict()
        #for i in range(classes):
            #temp_label = np.zeros(test_label.size)
            #label_index = np.where(test_label == (i + 1))
            #temp_label[label_index] = 1

            #fpr[i], tpr[i], _ = roc_curve(temp_label, decision_score[:, i])
            #roc_auc[i] = auc(fpr[i], tpr[i])

        # compute auc
        #auc_score = roc_auc_score(
            #test_label, log_reg.predict_proba(bold_test_subject), multi_class="ovr"
        #)

        # get predicted scores
        predict = log_reg.predict(bold_test_subject)
        predict_prob = log_reg.predict_proba(bold_test_subject)

        # set up confusion matrix
        true_values = np.asarray([np.sum(test_label == i) for i in [0, 1]])
        print(true_values)
        cm = (
            confusion_matrix(test_label, predict, labels=list([0, 1]))
            / true_values[:, None]
            * 100
        )

        scores.append(score)
        #aucs.append(auc_score)
        confusion_matrices.append(cm)

        evidence = 1.0 / (1.0 + np.exp(-log_reg.decision_function(bold_test_subject)))
        evidences.append(evidence)

        #roc_aucs.append(roc_auc)

        test_labels.append(test_label)
        pred_probs.append(predict_prob)
        decision_scores.append(decision_score)

    #roc_aucs = pd.DataFrame(data=roc_aucs)
    scores = np.asarray(scores)
    #aucs = np.asarray(aucs)
    confusion_matrices = np.stack(confusion_matrices)
    evidences = np.stack(evidences)

    test_labels = np.stack(test_labels)
    pred_probs = np.stack(pred_probs)
    decision_scores = np.stack(decision_scores)

    print(
        f"\nClassifier score: \n"
        f"scores: {scores.mean()} sd: {scores.std()}\n"
        #f"auc scores: {aucs.mean()} sd: {aucs.std()}\n"
        f"average confusion matrix:\n"
        f"{confusion_matrices.mean(axis=0)}"
    )

    return (
        scores,
        #aucs,
        confusion_matrices,
        evidences,
        test_labels,
        decision_scores,
        #roc_aucs,
    )

# %% set up function to run whole classification
def betweenSub_classification():
    print("Running between-subject classification...")
    # load data
    full_data = {}
    for sub in subIDs:
        full_data[f"{sub}"], _ = load_data(
            directory=data_path,
            subject_name=sub,
            mask_name="wholebrain",
            zscore_data=True,
        )

    # load labels
    stim_labels_allruns = get_shifted_labels(task="study", shift_size_TR=shift_TR)
    print(f"Label shape: {stim_labels_allruns.shape}")

    # set up train and test data
    bold_data, group_labels, subject_sample = betweenSub_operation_sample(
        full_data, stim_labels_allruns
    )

    # run model
    print("Running model...")

    (
        scores,
        #aucs,
        confusion_matrices,
        evidences,
        test_labels,
        decision_scores,
        #roc_aucs,
    ) = betweenSub_decode_model(bold_data, group_labels, subject_sample)

    mean_score = scores.mean()
    print(f"Mean score: {mean_score}")

    # save aucs for each operation to csv in results folder of working directory
    #aucs_df = pd.DataFrame(columns=["Maintain", "Replace", "Suppress", "sub"])
    #aucs_df["Maintain"] = roc_aucs.loc[:, 0]
    #aucs_df["Replace"] = roc_aucs.loc[:, 1]
    #aucs_df["Suppress"] = roc_aucs.loc[:, 2]
    #aucs_df["sub"] = [*range(0, 5)]

    # Create average confusion matrix figure
    avg_cm = confusion_matrices.mean(axis=0)

    avg_disp = ConfusionMatrixDisplay(
        confusion_matrix=avg_cm, display_labels=["Bad", "Good"]
    )
    avg_disp.plot(cmap=plt.cm.GnBu)
    plt.title(
        "Between-Subject Operation Classification \n (train on N-1 subs, test on held out sub)"
    )

    # Show the figures
    plt.show()

# %% run first model (classification of all operations)
betweenSub_classification()


# %% add suppress-forget count to participant memory results
filelist = [f for f in
            os.listdir('/Users/cnj678/Desktop/data_ML_final/memory_results')
            if
            f.endswith('.csv')]

for i in range(len(filelist)):
  filename = ('/Users/cnj678/Desktop/data_ML_final/memory_results/' + filelist[i])

  memory_data = pd.read_csv(filename)

  participant = memory_data.iloc[0]['subject']

  memory_data = memory_data.assign(Suppress_Forget=np.where(((memory_data['memory'] == 0) & (memory_data['condition_num'] == 3)), 1, 0))

  memory_data.to_csv(f'/Users/cnj678/Desktop/data_ML_final/sub_mem_data/{participant}_mem_data.csv', index=False)

# %% aggreagate memory results and find median for split
mem_path = '/Users/cnj678/Desktop/data_ML_final/sub_mem_data/*.csv'
mem_files = glob.glob(mem_path)
mem_data_merge = pd.concat([pd.read_csv(mem_file) for mem_file in mem_files],
                           ignore_index=True)

agg_select = ['Suppress_Forget']

agg_dict = {col: 'sum' for col in agg_select}
rename_dict = {col: col.lower() + '_sum' for col in agg_select}

data_agg = mem_data_merge.groupby(['subject']).agg(agg_dict).rename(columns=rename_dict).reset_index()


median_value = data_agg['suppress_forget_sum'].median()
print(median_value)



# %%
full_data = {}
    for sub in subIDs:
        full_data[f"{sub}"], _ = load_data(
            directory=data_path,
            subject_name=sub,
            mask_name="wholebrain",
            zscore_data=True,
        )

    # load labels
    stim_labels_allruns = get_shifted_labels(task="study", shift_size_TR=shift_TR)
    print(f"Label shape: {stim_labels_allruns.shape}")

    # set up train and test data
    bold_data, group_labels, subject_sample = betweenSub_operation_sample(
        full_data, stim_labels_allruns
    )

# %%
ps = PredefinedSplit(subject_sample)
for train, test in ps.split():
    train_data = bold_data[train]
    test_data = bold_data[test]
    train_label = group_labels[train]
    test_label = group_labels[test]


    model = LogisticRegression(penalty='l2' ,solver='lbfgs')


    model.fit(train_data, train_label)


    preds = model.predict(test_data)


    score = model.score(test_data, test_label)
    print(score)

    predict = model.predict(test_data)
    predict_prob = model.predict_proba(test_data)

    # set up confusion matrix
    true_values = np.asarray([np.sum(test_label == i) for i in [0,1]])
    print(true_values)
    cm = (
            confusion_matrix(test_label, predict, labels=list([0,1]))
            / true_values[:, None]
            * 100
    )
    print(cm)

# %%
ps = PredefinedSplit(subject_sample)
for train, test in ps.split():
    train_data = bold_data[train]
    test_data = bold_data[test]
    train_label = group_labels[train]
    test_label = group_labels[test]

    Fselect_fpr = SelectFpr(f_classif, alpha=0.05).fit(train_data, train_label)
    bold_train_subject = Fselect_fpr.transform(train_data)
    bold_test_subject = Fselect_fpr.transform(test_data)
    print(bold_train_subject.shape)
    print(bold_test_subject.shape)

    #model = LogisticRegression(penalty='l2' ,solver='lbfgs')

# %%
full_data = {}
    for sub in subIDs:
        full_data[f"{sub}"], _ = load_data(
            directory=data_path,
            subject_name=sub,
            mask_name="wholebrain",
            zscore_data=True,
        )

stim_labels_allruns = get_shifted_labels(task="study", shift_size_TR=shift_TR)

# %%
# this will restrict us to only suppress trials
labels_df = stim_labels_allruns
suppress_only = [3]
labels_df = labels_df[labels_df['condition'].isin(suppress_only)]
labels_df = labels_df.reset_index(drop=True)


operation_list = labels_df["condition"]
stim_onscreen = labels_df["stim_present"]
run_list = labels_df["run"]


# define operation period
operation_index = np.where(np.logical_or(stim_onscreen == 2, stim_onscreen == 3))[0]
rest_index = []
runs = run_list.unique()[1:]


operation_val = operation_list.values[operation_index]
run_val = run_list.values[operation_index]

# %%
# we want the first half of each run for feature selection
bold_data = []
operation_sample = operation_val

sub_loop = 0
for sub in subIDs:
    print(sub)
    bold = full_data[f"{sub}"][operation_index]

    if sub_loop == 0:
        bold_data = bold
        print(bold_data.shape)
        sub_loop = sub_loop + 1
    else:
        bold_data = np.concatenate((bold_data, bold))
        print(bold_data.shape)

subject_sample = np.repeat(range(0, 5), 300)
op_labels = np.tile(operation_sample, 5)

# %%
# here I am manually setting the group (good vs. bad) based on memory analyses at the end of the script
# 0: bad suppressor;  1: good suppresor
sub_004_group = np.ones(300)
sub_005_group = np.ones(300)
sub_006_group = np.zeros(300)
sub_024_group = np.ones(300)
sub_026_group = np.zeros(300)

sub_006_group[sub_006_group == 0] = 2
sub_026_group[sub_026_group == 0] = 2

group_labels = np.concatenate((sub_004_group, sub_005_group, sub_006_group, sub_024_group, sub_026_group))

# %%
print(bold_data.shape)
print(group_labels.shape)
print(subject_sample.shape)

print(op_labels)


# %%
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


scores = []
decision_scores = []
test_labels = []
pred_probs = []
confusion_matrices = []
# aucs = []
evidences = []
# roc_aucs = []

ps = PredefinedSplit(subject_sample)
for train, test in ps.split():
    train_data = bold_data[train]
    test_data = bold_data[test]
    train_label = group_labels[train]
    test_label = group_labels[test]

    # feature selection
    #Fselect_fpr = SelectFpr(f_classif, alpha=0.001).fit(train_data, train_label)
    #bold_train_subject = Fselect_fpr.transform(train_data)
    # bold_test_subject = Fselect_fpr.transform(test_data)

    svm = SVC(kernel='linear')  # You can choose different kernels: 'linear', 'rbf', 'poly', etc.
    svm.fit(train_data, train_label)

    # Predictions
    predictions = svm.predict(test_data)
    scores.append(accuracy_score(test_label, predictions))
    decision_scores.append(svm.decision_function(test_data))
    test_labels.append(test_label)
    #pred_probs.append(svm.predict_proba(test_data))
    confusion_matrices.append(confusion_matrix(test_label, predictions))
    evidences.append(svm.decision_function(test_data))

