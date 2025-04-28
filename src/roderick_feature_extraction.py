from sklearn.model_selection import train_test_split
from functions.feature_extraction import *
from functions.process_data import process_data


def main():
    # Process data
    data, target, scaler, numerical_cats = process_data('data/processed_data.csv')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, random_state=N, train_size=TRAIN_SIZE
    )

    # Choose one feature extraction method (or none)
    # perform_pca(data, X_train, X_test)
    perform_lda(X_train, y_train, X_test)
    # perform_lasso(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
