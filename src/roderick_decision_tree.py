import pandas.core.frame
import pandas.core.series
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from functions.process_data import *
from functions.feature_extraction import *
from functions.evaluation_metrics import *
from constants.constants import *

# Constants
MODEL_NAME = 'Decision Tree'
MAX_DEPTH = 25
MIN_IMPURITY_DECREASE = 0.00001
DIAGRAM_MAX_DEPTH = 3
DIAGRAM_FONT_SIZE = 4
DIAGRAM_DPI = 400


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
    X_test = align_columns(data, X_test)

    # Perform lasso regression
    # removed_features = ['DAYS_EMPLOYED', 'AGE', 'NAME_INCOME_TYPE_Student', 'NAME_FAMILY_STATUS_Separated',
    #                     'NAME_FAMILY_STATUS_Widow', 'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Managers',
    #                     'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Secretaries',
    #                     'OCCUPATION_TYPE_Waiters/barmen staff']  # Pre-run lasso regression results
    # # removed_features = perform_lasso(X_train, y_train, X_test, y_test)
    # X_train = X_train.drop(removed_features, axis=1, errors='ignore')
    # X_test = X_test.drop(removed_features, axis=1, errors='ignore')

    # Print index to feature
    print('Features list:')
    features = X_train.columns.tolist()
    for index in range(len(features)):
        print(f'{index}: {features[index]}')
    print()

    # Convert data back to numpy array
    if isinstance(X_train, pandas.core.frame.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(y_train, pandas.core.series.Series):
        y_train = y_train.to_numpy()
    if isinstance(X_test, pandas.core.frame.DataFrame):
        X_test = X_test.to_numpy()
    if isinstance(y_test, pandas.core.series.Series):
        y_test = y_test.to_numpy()

    # Create Decision Tree model
    tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_impurity_decrease=MIN_IMPURITY_DECREASE)

    # Train the model
    tree.fit(X_train, y_train)

    # Make predictions
    y_pred = tree.predict(X_test)
    y_train_pred = tree.predict(X_train)
    y_pred_prob = tree.predict_proba(X_test)[:, 1]

    # Get evaluation metrics
    train_score = accuracy_score(y_train, y_train_pred)
    roc_auc = plot_roc_curve(y_test, y_pred_prob, MODEL_NAME)
    pr_auc = plot_pr_curve(y_test, y_pred_prob, MODEL_NAME)
    eval_metrics = list(create_evaluation_metrics(y_test, y_pred))
    eval_metrics.append(roc_auc)
    eval_metrics.append(pr_auc)

    # Undo scaling
    T = tree.tree_
    thresh = T.threshold.copy()
    for node_id, feat_id in enumerate(T.feature):
        name = features[feat_id]

        if name in numerical_cats:
            index = numerical_cats.index(name)
            mu = scaler.mean_[index]
            sigma = scaler.scale_[index]
            # Invert scaling:  x_orig = x_scaled * sigma + mu
            T.threshold[node_id] = thresh[node_id] * sigma + mu

    # Get gini importances
    gini_importances = tree.feature_importances_
    df_gini_importances = (
        pd.Series(gini_importances, index=features)
        .sort_values(ascending=False)
        .head(10)
    )

    return tree, train_score, eval_metrics, features, df_gini_importances


def main():
    tree, train_score, eval_metrics, features, df_gini_importances = test_credit_approval_prediction()

    plot_tree(
        tree,
        max_depth=DIAGRAM_MAX_DEPTH,
        feature_names=features,
        class_names=["rejected", "approved"],
        filled=True,
        rounded=True,
        proportion=True,
        fontsize=DIAGRAM_FONT_SIZE
    )

    # Save graph as image
    current_directory_path = os.getcwd()
    subfolder_path = os.path.join(current_directory_path, 'graphs', 'Decision Tree')
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    file_path = os.path.join(subfolder_path, 'Decision Tree')
    plt.savefig(file_path, dpi=DIAGRAM_DPI)

    # Print evaluation metrics
    print(f'Training accuracy: {train_score}')
    accuracy, precision, recall, specificity, f1_score, confusion_matrix, roc_auc, pr_auc = eval_metrics
    print(f'Test accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall/Sensitivity: {recall}')
    print(f'Specificity: {specificity}')
    print(f'F1 score: {f1_score}')
    print('Confusion matrix:')
    print(confusion_matrix)
    print(f'ROC-AUC score: {roc_auc}')
    print(f'PR-AUC score: {pr_auc}')

    # Print insights
    print("\nTop 10 features by Gini importance:")
    print(df_gini_importances)


if __name__ == '__main__':
    main()
