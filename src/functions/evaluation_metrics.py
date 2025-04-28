import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt


def create_evaluation_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for index in range(len(y_true)):
        if y_pred[index] == 0 and y_true[index] == 0:
            tn += 1
        elif y_pred[index] == 0 and y_true[index] == 1:
            fn += 1
        elif y_pred[index] == 1 and y_true[index] == 0:
            fp += 1
        elif y_pred[index] == 1 and y_true[index] == 1:
            tp += 1
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    try:
        f1_score = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0
    try:
        specificity = tn / (tn + fp)
    except ZeroDivisionError:
        specificity = 0

    confusion_matrix = np.array([[tp, fp], [fn, tn]])

    return accuracy, precision, recall, specificity, f1_score, confusion_matrix


def plot_roc_curve(y_true, y_pred_prob, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob, pos_label=1)
    roc_auc = roc_auc_score(y_true, y_pred_prob)

    # Plot the ROC curve
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No skill')  # ROC curve for TPR = FPR (no skill)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")

    # Save graph as image
    current_directory_path = os.getcwd()
    subfolder_path = os.path.join(current_directory_path, 'graphs', model_name)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    file_path = os.path.join(subfolder_path, f'{model_name} ROC Curve')
    plt.savefig(file_path)

    return roc_auc


def plot_pr_curve(y_true, y_pred_prob, model_name):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)

    # Plot the PR curve
    plt.clf()
    plt.plot(recall, precision, label='PR curve (area = %0.2f)' % pr_auc)
    plt.plot([0, 1], [0, 0], 'k--', label='No skill')  # PR curve for precision = 0 (no skill)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} PR Curve')
    plt.legend(loc="lower right")

    # Save graph as image
    current_directory_path = os.getcwd()
    subfolder_path = os.path.join(current_directory_path, 'graphs', model_name)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    file_path = os.path.join(subfolder_path, f'{model_name} PR Curve')
    plt.savefig(file_path)

    return pr_auc


def create_accuracy_array(p_list, w_list, y_true, threshold):
    accuracy_array = []
    for p, w in zip(p_list, w_list):
        y_pred = (p @ w > threshold).astype(int)
        accuracy_array.append(accuracy_score(y_true, y_pred))

    return accuracy_array


def create_eval_metrics_array(p_list, w_list, y_true, threshold):
    accuracy_array = []
    precision_array = []
    recall_array = []
    f1_score_array = []
    specificity_array = []
    confusion_matrix_array = []

    for p, w in zip(p_list, w_list):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        y_pred = (p @ w > threshold).astype(int)
        accuracy_array.append(accuracy_score(y_true, y_pred))

        for index in range(len(y_true)):
            if y_pred[index] == 0 and y_true[index] == 0:
                tn += 1
            elif y_pred[index] == 0 and y_true[index] == 1:
                fn += 1
            elif y_pred[index] == 1 and y_true[index] == 0:
                fp += 1
            elif y_pred[index] == 1 and y_true[index] == 1:
                tp += 1
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0
        try:
            f1_score = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1_score = 0
        try:
            specificity = tn / (tn + fp)
        except ZeroDivisionError:
            specificity = 0

        precision_array.append(precision)
        recall_array.append(recall)
        f1_score_array.append(f1_score)
        specificity_array.append(specificity)
        confusion_matrix_array.append(np.array([[tp, fp], [fn, tn]]))

    return accuracy_array, precision_array, recall_array, specificity_array, f1_score_array, confusion_matrix_array


def create_y_pred_prob_array(p_list, w_list):
    y_pred_prob_array = []
    for p, w in zip(p_list, w_list):
        y_pred_prob = p @ w
        y_pred_prob_array.append(y_pred_prob)

    return y_pred_prob_array


def plot_roc_curve_array(y_true, y_pred_prob_array):
    plt.clf()
    roc_auc_array = []

    for order, y_pred_prob in enumerate(y_pred_prob_array):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob, pos_label=1)
        roc_auc = roc_auc_score(y_true, y_pred_prob)

        # Plot the ROC curve
        plt.plot(fpr, tpr, label=f'Order {order + 1}: ROC curve (area = %0.2f)' % roc_auc)
        roc_auc_array.append(roc_auc)

        # Find optimal threshold via Youden's J = TPR - FPR
        j_scores = tpr - fpr
        idx_opt = np.argmax(j_scores)
        thresh_opt = thresholds[idx_opt]
        fpr_opt, tpr_opt = fpr[idx_opt], tpr[idx_opt]

        # Mark the optimal point
        plt.scatter(fpr_opt, tpr_opt,
                    marker='o',
                    label=f'Order {order + 1}: Optimal Threshold = {thresh_opt:.2f}')

        # Annotate the threshold value
        plt.annotate(f'{thresh_opt:.2f}',
                     xy=(fpr_opt, tpr_opt),
                     xytext=(10, -10),
                     textcoords='offset points',
                     fontsize=8
                     )

    plt.plot([0, 1], [0, 1], 'k--', label='No skill')  # ROC curve for TPR = FPR (no skill)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Polynomial Ridge Regression ROC Curve')
    plt.legend(loc="lower right")

    # Save graph as image
    current_directory_path = os.getcwd()
    subfolder_path = os.path.join(current_directory_path, 'graphs', 'Polynomial Ridge Regression')
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    file_path = os.path.join(subfolder_path, f'Polynomial Ridge Regression ROC Curve')
    plt.savefig(file_path)

    return roc_auc_array


def plot_pr_curve_array(y_true, y_pred_prob_array):
    pr_auc_array = []
    plt.clf()
    for order, y_pred_prob in enumerate(y_pred_prob_array):
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = auc(recall, precision)

        # Plot the PR curve
        plt.plot(recall, precision, label=f'Order {order + 1}: PR curve (area = %0.2f)' % pr_auc)
        pr_auc_array.append(pr_auc)

    plt.plot([0, 1], [0, 0], 'k--', label='No skill')  # PR curve for precision = 0 (no skill)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Polynomial Ridge Regression PR Curve')
    plt.legend(loc="lower right")

    # Save graph as image
    current_directory_path = os.getcwd()
    subfolder_path = os.path.join(current_directory_path, 'graphs', 'Polynomial Ridge Regression')
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    file_path = os.path.join(subfolder_path, f'Polynomial Ridge Regression PR Curve')
    plt.savefig(file_path)

    return pr_auc_array


def create_accuracy_array_automated(ridge_list, Ptrain_list, y_true, threshold):
    accuracy_array = []
    for ridge, Ptrain in zip(ridge_list, Ptrain_list):
        y_pred = (ridge.predict(Ptrain) > threshold).astype(int)
        accuracy_array.append(accuracy_score(y_true, y_pred))

    return accuracy_array


def create_eval_metrics_array_automated(ridge_list, Ptest_list, y_true, threshold):
    accuracy_array = []
    precision_array = []
    recall_array = []
    f1_score_array = []
    specificity_array = []
    confusion_matrix_array = []

    for ridge, Ptest in zip(ridge_list, Ptest_list):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        y_pred = (ridge.predict(Ptest) > threshold).astype(int)
        accuracy_array.append(accuracy_score(y_true, y_pred))

        for index in range(len(y_true)):
            if y_pred[index] == 0 and y_true[index] == 0:
                tn += 1
            elif y_pred[index] == 0 and y_true[index] == 1:
                fn += 1
            elif y_pred[index] == 1 and y_true[index] == 0:
                fp += 1
            elif y_pred[index] == 1 and y_true[index] == 1:
                tp += 1
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0
        try:
            f1_score = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1_score = 0
        try:
            specificity = tn / (tn + fp)
        except ZeroDivisionError:
            specificity = 0

        precision_array.append(precision)
        recall_array.append(recall)
        f1_score_array.append(f1_score)
        specificity_array.append(specificity)
        confusion_matrix_array.append(np.array([[tp, fp], [fn, tn]]))

    return accuracy_array, precision_array, recall_array, specificity_array, f1_score_array, confusion_matrix_array
