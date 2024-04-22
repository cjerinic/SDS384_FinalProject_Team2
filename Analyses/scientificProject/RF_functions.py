from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFpr
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import numpy as np
from sklearn.model_selection import PredefinedSplit

def betweenSub_decode_model_RF(bold_data, op_labels, subject_sample, m_depth):
    scores = []
    decision_scores = []
    test_labels = []
    pred_probs = []
    confusion_matrices = []
    aucs = []
    evidences = []
    roc_aucs = []


    ps = PredefinedSplit(subject_sample)
    for train, test in ps.split():
        train_data = bold_data[train]
        test_data = bold_data[test]
        train_label = op_labels[train]
        test_label = op_labels[test]

        # Feature selection
        Fselect_fpr = SelectFpr(f_classif, alpha=0.001).fit(train_data, train_label)
        bold_train_subject = Fselect_fpr.transform(train_data)
        bold_test_subject = Fselect_fpr.transform(test_data)

        # Train with Random Forest Classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, max_depth = m_depth)
        rf_classifier.fit(bold_train_subject, train_label)

        # Now test on the held out subject
        score = rf_classifier.score(bold_test_subject, test_label)
        decision_score = rf_classifier.predict_proba(bold_test_subject)

        classes = np.unique(test_label).size

        # Calculate AUC for each operation
        roc_auc = {}
        for i in range(classes):
            temp_label = np.zeros(test_label.size)
            label_index = np.where(test_label == (i + 1))
            temp_label[label_index] = 1

            fpr, tpr, _ = roc_curve(temp_label, decision_score[:, i])
            roc_auc[i] = auc(fpr, tpr)

        # Compute AUC
        auc_score = roc_auc_score(
            test_label, decision_score, multi_class="ovr"
        )

        # Set up confusion matrix
        true_values = np.asarray([np.sum(test_label == i) for i in [1, 2, 3]])
        cm = (
            confusion_matrix(test_label, rf_classifier.predict(bold_test_subject), labels=list([1, 2, 3]))
            / true_values[:, None]
            * 100
        )

        scores.append(score)
        aucs.append(auc_score)
        confusion_matrices.append(cm)
        roc_aucs.append(roc_auc)

        test_labels.append(test_label)
        pred_probs.append(decision_score)

    roc_aucs = pd.DataFrame(data=roc_aucs)
    scores = np.asarray(scores)
    aucs = np.asarray(aucs)
    confusion_matrices = np.stack(confusion_matrices)

    test_labels = np.stack(test_labels)
    pred_probs = np.stack(pred_probs)

    print(
        f"\nClassifier score: \n"
        f"scores: {scores.mean()} sd: {scores.std()}\n"
        f"auc scores: {aucs.mean()} sd: {aucs.std()}\n"
        f"average confusion matrix:\n"
        f"{confusion_matrices.mean(axis=0)}"
    )

    return (
        scores,
        aucs,
        confusion_matrices,
        evidences,
        test_labels,
        decision_scores,
        roc_aucs,
    )

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
    bold_data, op_labels, subject_sample = betweenSub_operation_sample(
        full_data, stim_labels_allruns
    )

    # run model
    print("Running model...")

    (
        scores,
        aucs,
        confusion_matrices,
        evidences,
        test_labels,
        decision_scores,
        roc_aucs,
    ) = betweenSub_decode_model_RF(bold_data, op_labels, subject_sample, 10)

    mean_score = scores.mean()
    print(f"Mean score: {mean_score}")

    # save aucs for each operation to csv in results folder of working directory
    aucs_df = pd.DataFrame(columns=["Maintain", "Replace", "Suppress", "sub"])
    aucs_df["Maintain"] = roc_aucs.loc[:, 0]
    aucs_df["Replace"] = roc_aucs.loc[:, 1]
    aucs_df["Suppress"] = roc_aucs.loc[:, 2]
    aucs_df["sub"] = [*range(0, 5)]
    aucs_df.to_csv('/Users/owenfriend/Downloads/SDS384_FinalProjectData_Team2/aucs_df.csv')
    # Create average confusion matrix figure
    avg_cm = confusion_matrices.mean(axis=0)

    avg_disp = ConfusionMatrixDisplay(
        confusion_matrix=avg_cm, display_labels=["Maintain", "Replace", "Suppress"]
    )
    avg_disp.plot(cmap=plt.cm.GnBu)
    plt.title(
        "Between-Subject Operation Classification \n (train on N-1 subs, test on held out sub)"
    )

    # Show the figures
    plt.show()