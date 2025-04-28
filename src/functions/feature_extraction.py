import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
from constants.constants import *


# Constants
VARIANCE_RETAINED = 0.95


def perform_pca(data, X_train, X_test):
    # Perform PCA
    pca = PCA()
    pca.fit(data)
    pc_values = np.arange(pca.n_components_) + 1

    # Plot cumulative explained variance
    plt.clf()
    plt.plot(pc_values, pca.explained_variance_ratio_.cumsum(), "o-")
    plt.title("Cumulative Explained Variance against Number of Principal Components")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")

    # Save graph as image
    current_directory_path = os.getcwd()
    subfolder_path = os.path.join(current_directory_path, 'graphs', 'PCA')
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    file_path = os.path.join(subfolder_path, 'PCA Cumulative Explained Variance')
    plt.savefig(file_path)

    # Plot scree plot
    plt.clf()
    plt.plot(pc_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')

    # Save graph as image
    current_directory_path = os.getcwd()
    subfolder_path = os.path.join(current_directory_path, 'graphs', 'PCA')
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    file_path = os.path.join(subfolder_path, 'PCA Scree Plot')
    plt.savefig(file_path)

    # Apply PCA to data
    pca = PCA(VARIANCE_RETAINED)
    pca.fit(data)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test


def perform_lda(X_train, y_train, X_test):
    # Perform LDA
    lda = LinearDiscriminantAnalysis()

    # Apply LDA to data
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)

    # Plot LDA graphs
    plt.yticks([])  # Hide y-axis ticks
    approved_label = 'Label = 1 (Approved)'
    rejected_label = 'Label = 0 (Rejected)'
    plt.xlabel('LD1')
    plt.ylim(-1, 1)

    # Plot LD1 for rejected class
    plt.clf()
    plt.title('LDA: Application Record Data Projected onto LD1 (Rejected)')
    plt.scatter(X_train[y_train == 0, 0], np.zeros(len(y_train[y_train == 0])), c='red', alpha=0.7,
                label=rejected_label)
    plt.legend()

    # Save graph as image
    current_directory_path = os.getcwd()
    subfolder_path = os.path.join(current_directory_path, 'graphs', 'LDA')
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    file_path = os.path.join(subfolder_path, 'LDA (Rejected)')
    plt.savefig(file_path)

    # Plot LD1 for approved class
    plt.clf()
    plt.title('LDA: Application Record Data Projected onto LD1 (Approved)')
    plt.scatter(X_train[y_train == 1, 0], np.zeros(len(y_train[y_train == 1])), c='green', alpha=0.7,
                label=approved_label)
    plt.legend()

    # Save graph as image
    current_directory_path = os.getcwd()
    subfolder_path = os.path.join(current_directory_path, 'graphs', 'LDA')
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    file_path = os.path.join(subfolder_path, 'LDA (Approved)')
    plt.savefig(file_path)

    # Plot LD1 for both classes
    plt.clf()
    plt.title('LDA: Application Record Data Projected onto LD1')
    plt.scatter(X_train[y_train == 0, 0], np.zeros(len(y_train[y_train == 0])), c='red', alpha=0.7,
                label=rejected_label)
    plt.scatter(X_train[y_train == 1, 0], np.zeros(len(y_train[y_train == 1])), c='green', alpha=0.7,
                label=approved_label)
    plt.legend()
    # Save graph as image
    current_directory_path = os.getcwd()
    subfolder_path = os.path.join(current_directory_path, 'graphs', 'LDA')
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    file_path = os.path.join(subfolder_path, 'LDA')
    plt.savefig(file_path)

    return X_train, y_train, X_test


def perform_lasso(X_train, y_train, X_test, y_test):
    # Perform 5-fold cross-validation to determine best alpha value for maximum R² score
    lasso = LassoCV(cv=5, alphas=np.logspace(-6, 2, 50), max_iter=1000, random_state=N)
    lasso.fit(X_train, y_train)
    print(f'Optimal alpha value: {lasso.alpha_}')

    # Print R² scores
    train_score = lasso.score(X_train, y_train)
    test_score = lasso.score(X_test, y_test)
    print(f"Lasso regression training R² score: {train_score}")
    print(f"Lasso regression test R² score: {test_score}")

    # Plot graph
    coefficients = pd.Series(lasso.coef_, index=X_train.columns).sort_values(ascending=True)
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 6))
    coefficients.plot(kind='bar', ax=ax)
    ax.set_xticklabels(coefficients.index, rotation=45, ha='right')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.45)
    plt.xlabel('Feature')
    plt.ylabel('Coefficient')
    plt.title('Lasso Coefficients by Feature')

    # Save graph as image
    current_directory_path = os.getcwd()
    subfolder_path = os.path.join(current_directory_path, 'graphs', 'Lasso')
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    file_path = os.path.join(subfolder_path, 'Lasso Regression Features')
    plt.savefig(file_path)

    # Reset plot
    plt.close()

    # Get list of features removed by lasso regression
    removed_features = X_train.columns[lasso.coef_ == 0].tolist()

    return removed_features
