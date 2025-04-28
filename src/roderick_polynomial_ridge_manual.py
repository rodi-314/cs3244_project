import pandas as pd
import pandas.core.frame
import pandas.core.series
from sklearn.model_selection import train_test_split
from functions.process_data import *
from functions.feature_extraction import *
from functions.polynomial_regression import *
from functions.evaluation_metrics import *

# Constants
MAX_ORDER = 3
THRESHOLD = 0.78
REG = 0.1
BEST_ORDER = 3
NO_OF_TERMS = 8


def test_credit_approval_prediction():
    # Imbalanced dataset
    data, target, scaler, numerical_cats = process_data('data/processed_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, random_state=N, train_size=TRAIN_SIZE, stratify=target
    )

    # Balanced dataset
    # X_train, y_train, scaler, numerical_cats = process_data('data/resampled_data_smote_82percent.csv')
    # X_test, y_test, scaler, numerical_cats = process_data('data/test_set_percent82_smote.csv')

    # Align columns and save feature names
    X_train = align_columns(data, X_train)
    original_features = X_train.columns.tolist()
    X_test = align_columns(data, X_test)

    # Choose one feature extraction method (or none)
    # PCA
    # X_train, X_test = perform_pca(data, X_train, X_test)
    # original_features = [f"PD{i+1}" for i in range(X_train.shape[1])]
    # LDA
    # X_train, y_train, X_test = perform_lda(X_train, y_train, X_test)
    # original_features = [f"LD{i + 1}" for i in range(X_train.shape[1])]
    # Lasso Regression
    removed_features = ['DAYS_EMPLOYED', 'AGE', 'NAME_INCOME_TYPE_Student', 'NAME_FAMILY_STATUS_Separated',
                        'NAME_FAMILY_STATUS_Widow', 'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Managers',
                        'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Secretaries',
                        'OCCUPATION_TYPE_Waiters/barmen staff']  # Pre-run lasso regression results
    # removed_features = perform_lasso(X_train, y_train, X_test, y_test)
    X_train = X_train.drop(removed_features, axis=1, errors='ignore')
    X_test = X_test.drop(removed_features, axis=1, errors='ignore')
    original_features = X_train.columns.tolist()

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
    w_list = create_w_list(Ptrain_list, y_train, REG)
    train_score = create_accuracy_array(Ptrain_list, w_list, y_train, THRESHOLD)

    y_pred_prob_array = create_y_pred_prob_array(Ptest_list, w_list)
    roc_auc = plot_roc_curve_array(y_test, y_pred_prob_array)
    pr_auc = plot_pr_curve_array(y_test, y_pred_prob_array)

    eval_metrics = list(create_eval_metrics_array(Ptest_list, w_list, y_test, THRESHOLD))
    eval_metrics.append(roc_auc)
    eval_metrics.append(pr_auc)

    # Regenerate feature names
    poly = PolynomialFeatures(degree=BEST_ORDER,
                              include_bias=False)
    poly.fit(X_train)  # only to build names
    feat_names = poly.get_feature_names_out(original_features)

    # Grab the corresponding weights
    coefs = w_list[BEST_ORDER - 1].flatten()[1:]  # Remove bias column

    # Rank and display top NO_OF_TERMS
    coef_df = pd.Series(coefs, index=feat_names)

    # Show sign of terms as well
    top_idx = coef_df.abs().nlargest(NO_OF_TERMS).index
    signed_top = coef_df.loc[top_idx]
    top_terms = pd.DataFrame({
        'coefficient': signed_top,
        'sign': np.sign(signed_top)
    })
    top_terms = top_terms.reindex(top_idx)

    top_coeffs = coef_df.loc[top_idx]

    # Plot graph and save
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 6))
    top_coeffs.sort_values().plot(kind='bar', ax=ax)
    ax.set_ylabel('Coefficient Value')
    ax.set_title(f'Top {NO_OF_TERMS} Polynomial Terms by Coefficient')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out_dir = os.path.join(os.getcwd(), 'graphs', 'Polynomial Ridge Regression')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'Top Terms Coefficients')
    plt.savefig(out_path)

    return w_list, train_score, eval_metrics, top_terms.to_string()


def main():
    w_list, train_score, eval_metrics, top_terms = test_credit_approval_prediction()
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
    print(f'PR-AUC score: {pr_auc_array}\n')
    print(f"Top {NO_OF_TERMS} polynomial terms by |weight|:")
    print(top_terms)


if __name__ == '__main__':
    main()
