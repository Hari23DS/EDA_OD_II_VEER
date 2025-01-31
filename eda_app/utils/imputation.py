import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def auto_impute(df):
    """
    Automatically imputes missing values using SimpleImputer and RandomForest.
    """
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Step 1: Impute missing numerical values with median
    num_imputer = SimpleImputer(strategy="median")
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Step 2: Impute missing categorical values with most frequent
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Step 3: Impute numerical columns using Random Forest
    def impute_numerical(df, num_cols):
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                train = df[df[col].notnull()]
                test = df[df[col].isnull()]
                predictors = [c for c in num_cols if c != col]

                if test.empty:
                    continue  # Skip if no missing values

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(train[predictors], train[col])
                df.loc[df[col].isnull(), col] = model.predict(test[predictors])
        return df
    
    df = impute_numerical(df, num_cols)

    # Step 4: Impute categorical columns using Random Forest
    def impute_categorical(df, cat_cols):
        label_encoders = {}
        for col in cat_cols:
            df[col] = df[col].astype(str)  # Convert to string to handle encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

            train = df[df[col] != -1]
            test = df[df[col] == -1]
            predictors = [c for c in cat_cols if c != col]

            if not test.empty:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(train[predictors], train[col])
                df.loc[df[col] == -1, col] = model.predict(test[predictors])
            
            df[col] = le.inverse_transform(df[col].astype(int))
        return df, label_encoders
    
    df, label_encoders = impute_categorical(df, cat_cols)
    return df

def impute_missing_values(df, method=1):
    """
    Function to impute missing values in a DataFrame.
    Supports:
    - Basic Imputation (Method 1)
    - MICE (Method 2)
    """
    if method == 1:
        # Basic Imputation
        df_basic_imp = df.copy()
        numerical_cols = df_basic_imp.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df_basic_imp.select_dtypes(include=['object']).columns

        def has_outliers(column):
            """Check if a numerical column has outliers using IQR."""
            Q1, Q3 = column.quantile(0.25), column.quantile(0.75)
            IQR = Q3 - Q1
            return ((column < (Q1 - 1.5 * IQR)) | (column > (Q3 + 1.5 * IQR))).any()

        for col in numerical_cols:
            if df_basic_imp[col].isnull().any():
                if has_outliers(df_basic_imp[col].dropna()):
                    df_basic_imp[col] = df_basic_imp[col].fillna(df_basic_imp[col].median())
                else:
                    df_basic_imp[col] = df_basic_imp[col].fillna(df_basic_imp[col].mean())

        for col in categorical_cols:
            if df_basic_imp[col].isnull().any():
                df_basic_imp[col] = df_basic_imp[col].fillna(df_basic_imp[col].mode()[0])

        return df_basic_imp

    elif method == 2:
        # MICE (Iterative Imputer)
        original_columns, original_dtypes = df.columns, df.dtypes
        df_num, df_cat = df.select_dtypes(include=np.number), df.select_dtypes(exclude=np.number)
        mappings, encoded_data = {}, {}

        # Encode categorical variables before imputation
        for col in df_cat.columns:
            unique_values = df[col].dropna().unique()
            if len(unique_values) <= 10:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(df[[col]])
                encoded_columns = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                mappings[col] = {'type': 'one_hot', 'encoder': encoder, 'columns': encoded_columns}
                encoded_data[col] = pd.DataFrame(encoded, columns=encoded_columns, index=df.index)
            else:
                freq_map = df[col].value_counts(normalize=True).to_dict()
                encoded = df[col].map(freq_map)
                mappings[col] = {'type': 'frequency', 'map': freq_map}
                encoded_data[col] = encoded

        df_encoded = pd.concat([df_num] + [encoded_data[col] for col in df_cat.columns], axis=1)

        # Perform MICE (Multiple Imputation by Chained Equations)
        imputer = IterativeImputer(random_state=0)
        df_adv_imp = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)

        # Decode categorical variables after imputation
        for col, meta in mappings.items():
            if meta['type'] == 'one_hot':
                max_col = df_adv_imp[meta['columns']].idxmax(axis=1)
                df_adv_imp[col] = max_col.str.replace(f"{col}_", "")
                df_adv_imp.drop(columns=meta['columns'], inplace=True)
            elif meta['type'] == 'frequency':
                df_adv_imp[col] = df_adv_imp[col].map({v: k for k, v in meta['map'].items()})

        df_adv_imp = df_adv_imp[original_columns]

        # Ensure correct data types after imputation
        for col in df_adv_imp.columns:
            df_adv_imp[col] = df_adv_imp[col].astype(original_dtypes[col])

        return df_adv_imp

    else:
        raise ValueError("Invalid method selection. Choose 1 for Basic or 2 for MICE.")

