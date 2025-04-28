import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Constants
VARIANCE_RETAINED = 0.95
TRAIN_SIZE = 0.8
N = 5


def align_columns(df_ref, df_to_align):
    """
    Return a copy of df_to_align whose columns are ordered to match df_ref.
    Any extra columns in df_to_align are dropped; any missing ones become NaN.
    """
    return df_to_align.reindex(columns=df_ref.columns)


def process_data(filepath):
    df = pd.read_csv(filepath)

    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(df)

    # Drop label and ID columns
    df = df.drop(['ID', 'NAME_HOUSING_TYPE_GROUPED'], axis=1, errors='ignore')

    # Display the first few rows of the features
    print("\nFirst few rows of the features:")
    print(df)

    # Classifying the columns into categorical and numerical
    # Binary categorical (to label encode)
    binary_cats = ['CODE_GENDER']  # M/F -> 0/1

    # Multi-category (to one-hot encoding)
    multi_cats = ['NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'OCCUPATION_TYPE',
                  'Grouped_Housing_Type']

    # Already numerical or binary (no encoding needed)
    numerical_or_binary_cats = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
                                'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
                                'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL',
                                'CNT_FAM_MEMBERS', 'AGE', 'DAYS_EMPLOYED_CLEAN']

    # Numerical categories to be scaled
    numerical_cats = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'AGE',
                      'DAYS_EMPLOYED_CLEAN']

    # Encode ordinal categories (just education type)
    education_mapping = [['Lower secondary', 'Secondary / secondary special',
                          'Incomplete higher', 'Higher education', 'Academic degree']]
    ordinal_encoder = OrdinalEncoder(categories=education_mapping)
    df["NAME_EDUCATION_TYPE"] = ordinal_encoder.fit_transform(df[["NAME_EDUCATION_TYPE"]])

    # Encode Binary Categories
    # Check the unique values in 'CODE_GENDER'
    print(f"\nUnique values in {binary_cats[0]}: {df[binary_cats[0]].unique()}")

    # Assuming the values are 'F' and 'M', map them to 0 and 1
    # Adjust the dictionary {'F': 0, 'M': 1} if actual values are different
    gender_map = {'F': 0, 'M': 1}  # Example map, adjust if needed
    df['CODE_GENDER_encoded'] = df['CODE_GENDER'].map(gender_map)

    # Drop the original 'CODE_GENDER' column if only the encoded version is required
    df = df.drop(columns=['CODE_GENDER'])

    print("DataFrame after binary encoding 'CODE_GENDER':")
    print(df)

    # Encode Multi-Category Columns using One-Hot Encoding
    print(f"\nOriginal shape before one-hot encoding: {df.shape}")
    print(f"Columns to be one-hot encoded: {multi_cats}")

    df = pd.get_dummies(df,
                        columns=multi_cats,
                        prefix=multi_cats,  # Uses column name as prefix for new columns
                        prefix_sep='_',  # Separator (e.g., NAME_INCOME_TYPE_Student)
                        drop_first=True,  # Drops one category per feature to avoid multicollinearity
                        dummy_na=False)  # Set to True if you want an explicit column for NaN values

    print(f"Shape after one-hot encoding: {df.shape}")
    print("DataFrame head after one-hot encoding (showing some original and new columns):")
    print(df)

    # Display columns after encoding
    print("Columns after all encoding:")
    print(df.columns.tolist())

    # Scale all numeric columns using StandardScaler
    scaler = StandardScaler()
    df[numerical_cats] = scaler.fit_transform(df[numerical_cats])
    print('\nDataFrame after scaling:')
    print(df)

    # Drop label column
    target = df['label']
    data = df.drop(['label'], axis=1)
    print('\nDataFrame after dropping label column:')
    print(data)

    # Convert the cleaned and scaled DataFrame back to a NumPy array
    # data = df.to_numpy()
    # target = target.to_numpy()
    print('\nLabels:')
    print(target)
    print()

    return data, target, scaler, numerical_cats
