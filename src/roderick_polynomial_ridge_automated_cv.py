import pandas.core.frame
import pandas.core.series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from functions.process_data import *
from functions.feature_extraction import *
from functions.polynomial_regression import *
from functions.evaluation_metrics import *


# Constants
MAX_ORDER = 3
THRESHOLD = 0.5


def test_credit_approval_prediction():
    # Imbalanced dataset
    data, target, scaler, numerical_cats = process_data('data/processed_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, random_state=N, train_size=TRAIN_SIZE, stratify=target
    )

    # Balanced dataset
    X_train, y_train, scaler, numerical_cats = process_data('data/resampled_data_smote_82percent.csv')
    X_test, y_test, scaler, numerical_cats = process_data('data/test_set_percent82_smote.csv')

    # Align columns and save feature names
    X_train = align_columns(data, X_train)
    X_test = align_columns(data, X_test)

    # Choose one feature extraction method (or none)
    # PCA
    X_train, X_test = perform_pca(data, X_train, X_test)
    # LDA
    # X_train, y_train, X_test = perform_lda(X_train, y_train, X_test)
    # original_features = [f"LD{i + 1}" for i in range(X_train.shape[1])]
    # Lasso Regression
    # removed_features = ['DAYS_EMPLOYED', 'AGE', 'NAME_INCOME_TYPE_Student', 'NAME_FAMILY_STATUS_Separated',
    #                     'NAME_FAMILY_STATUS_Widow', 'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Managers',
    #                     'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Secretaries',
    #                     'OCCUPATION_TYPE_Waiters/barmen staff']  # Pre-run lasso regression results
    # # removed_features = perform_lasso(X_train, y_train, X_test, y_test)
    # X_train = X_train.drop(removed_features, axis=1, errors='ignore')
    # X_test = X_test.drop(removed_features, axis=1, errors='ignore')

    # Convert data back to numpy array
    if isinstance(X_train, pandas.core.frame.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(y_train, pandas.core.series.Series):
        y_train = y_train.to_numpy()
    if isinstance(X_test, pandas.core.frame.DataFrame):
        X_test = X_test.to_numpy()
    if isinstance(y_test, pandas.core.series.Series):
        y_test = y_test.to_numpy()

    Ptrain_list = create_P_list(X_train, MAX_ORDER)
    Ptest_list = create_P_list(X_test, MAX_ORDER)

    ridge_list = []
    y_pred_prob_array = []
    for order, Ptrain_Ptest in enumerate(zip(Ptrain_list, Ptest_list)):
        Ptrain, Ptest = Ptrain_Ptest

        # Perform 5-fold cross-validation to determine best alpha value for maximum R² score
        ridge = RidgeCV(cv=5, alphas=(0.0001, 0.001, 0.001, 0.1, 1, 10, 100, 1000, 10000))
        ridge.fit(Ptrain, y_train)
        print(f'Order {order + 1}')
        print(f'Optimal alpha value: {ridge.alpha_}')

        # Print R² scores
        train_score = ridge.score(Ptrain, y_train)
        test_score = ridge.score(Ptest, y_test)
        print(f"Ridge regression training R² score: {train_score}")
        print(f"Ridge regression test R² score: {test_score}\n")

        y_pred_prob = ridge.predict(Ptest)
        y_pred_prob_array.append(y_pred_prob)
        ridge_list.append(ridge)

    train_score = create_accuracy_array_automated(ridge_list, Ptrain_list, y_train, THRESHOLD)

    roc_auc = plot_roc_curve_array(y_test, y_pred_prob_array)
    pr_auc = plot_pr_curve_array(y_test, y_pred_prob_array)

    eval_metrics = list(create_eval_metrics_array_automated(ridge_list, Ptest_list, y_test, THRESHOLD))
    eval_metrics.append(roc_auc)
    eval_metrics.append(pr_auc)

    return ridge_list, train_score, eval_metrics


def main():
    w_list, train_score, eval_metrics = test_credit_approval_prediction()
    # print(w_list)
    print(f'Training accuracy: {train_score}')
    (accuracy_array, precision_array, recall_array, specificity_array, f1_score_array, confusion_matrix_array,
     roc_auc_array, pr_auc_array) = eval_metrics
    print(f'Test accuracy: {accuracy_array}')
    print(f'Precision: {precision_array}')
    print(f'Recall/Sensitivity: {recall_array}')
    print(f'Specificity: {specificity_array}')
    print(f'F1 score: {f1_score_array}')
    for order, confusion_matrix_order in enumerate(confusion_matrix_array):
        print(f'Confusion matrix for order {order + 1}:')
        print(confusion_matrix_order)
    print(f'ROC-AUC score: {roc_auc_array}')
    print(f'PR-AUC score: {pr_auc_array}')


if __name__ == '__main__':
    main()
