import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

class Imputer:
    static_cols = [
            'stay', 'day', 'old_extubation_status', 'Patient_id', 'Patient_id_2',
            'patient_day', 'patient', 'extubation_status', 'trach_collar_status',
            'labels', 'next_day_extubation_status', 'Norepinephrine_flag', 'Intubation_flag']

    @staticmethod
    def binning_imputation(train_data, val_data, test_data):

        # features encoded as 0 or 1
        binary_cols = []
        for col in train_data.columns:
            num_cats = len(train_data[col].value_counts().index)
            if num_cats <= 2:  # if two or less unique values, this is binary
                binary_cols.append(col)

        # features that are categorical (scored on a scale)
        categorical_cols = [
            x for x in train_data.columns
            if x.startswith('GCS') or x == 'RASS_score' or x == 'Norepinephrine_rate'
        ]
        
        ignore_cols = Imputer.static_cols.copy()
        ignore_cols.append('labels')
        def bin_boundaries(n):
            return [i / n for i in range(1, n)]

        # make columns following quantile bins
        def quantile_class(value, low_bound, high_bound):
            if pd.isna(value):
                return 0
            elif value >= low_bound and value < high_bound:
                return 1
            else:
                return 0

        # binning encoding for train set
        def encode_train(train_data, num_bins):
            quantiles_dict = {}
            quantiles = bin_boundaries(num_bins)

            binning_df = train_data.copy()

            for col in train_data.columns:
                if col not in ignore_cols:  # skip descriptive columns that may be used later on for modeling
                    if col in binary_cols:
                        # if categorical column, then make another column encoding if the data is missing or not
                        binning_df[f'{col}_mask'] = [
                            1 if pd.isna(x) else 0 for x in binning_df[col]
                        ]

                        # make column for each label
                        for val in [0, 1]:
                            binning_df[f'{col}_{val}'] = [
                                1 if x == val else 0 for x in binning_df[col]
                            ]
                        binning_df = binning_df.drop(columns=[col])
                    elif col in categorical_cols:
                        # categorical column, add mask feature then make bins for each integer score
                        binning_df[f'{col}_mask'] = [
                            1 if pd.isna(x) else 0 for x in binning_df[col]
                        ]
                        unique_vals = np.unique(
                            train_data[~pd.isna(train_data[col])][col].values)
                        valid_scores = unique_vals[unique_vals.astype(int) ==
                                                unique_vals]
                        i = 0
                        while i < len(valid_scores) - 1:
                            new_col = f"{col}_[{valid_scores[i]},{valid_scores[i+1]})"
                            binning_df[new_col] = \
                            [1 if (x >= valid_scores[i] and x < valid_scores[i+1]) else 0 for x in binning_df[col]]
                            i += 1
                        binning_df = binning_df.drop(columns=[col])
                    else:
                        # numerical column, add mask feature then proceed with Binning bins
                        binning_df[f'{col}_mask'] = [
                            1 if pd.isna(x) else 0 for x in binning_df[col]
                        ]

                        # get non-missing feature values acros whole dataframe
                        # --> compute bin boundaries and save in a dictionary
                        vals = binning_df[~pd.isna(binning_df[col])][col].values
                        quantile_values = np.quantile(vals, quantiles)
                        if len(quantile_values) != len(np.unique(quantile_values)):
                            print(f"{col} has duplicate bins")
                            break
                        quantiles_dict[col] = quantile_values

                        # save first bin starting from -inf (just to be safe)
                        binning_df[f'{col}_[{-np.inf}, {quantile_values[0]})'] = \
                        [quantile_class(x, -np.inf, quantile_values[0]) for x in binning_df[col]]

                        # use for loop to encode all the inner bins
                        if num_bins > 2:
                            for i in range(0, num_bins - 2):
                                binning_df[f'{col}_[{quantile_values[i]}, {quantile_values[i+1]})'] = \
                                [quantile_class(x, quantile_values[i], quantile_values[i+1]) for x in binning_df[col]]

                        # extend last bin to inf (just to be safe)
                        binning_df[f'{col}_[{quantile_values[-1]}, {np.inf}]'] = \
                        [quantile_class(x, quantile_values[-1], np.inf) for x in binning_df[col]]

                        # remove original numerical feature column
                        binning_df = binning_df.drop(columns=[col])
            return binning_df, quantiles_dict

        # binning encoding for test set
        def encode_test(train_data, test_data, quantiles_dict, binning_df):
            new_df = test_data.copy()

            for col in new_df.columns:
                if col not in ignore_cols:  # skip descriptive columns that may be used later on for modeling
                    if col in binary_cols:
                        # if categorical column, then make another column encoding if the data is missing or not
                        new_df[f'{col}_mask'] = [
                            1 if pd.isna(x) else 0 for x in new_df[col]
                        ]

                        # make column for each label
                        for val in [0, 1]:
                            new_df[f'{col}_{val}'] = [
                                1 if x == val else 0 for x in new_df[col]
                            ]
                        new_df = new_df.drop(columns=[col])
                    elif col in categorical_cols:
                        # categorical column, add mask feature then make bins for each integer score
                        new_df[f'{col}_mask'] = [
                            1 if pd.isna(x) else 0 for x in new_df[col]
                        ]
                        unique_vals = np.unique(
                            train_data[~pd.isna(train_data[col])][col].values)
                        valid_scores = unique_vals[unique_vals.astype(int) ==
                                                unique_vals]
                        i = 0
                        while i < len(valid_scores) - 1:
                            new_col = f"{col}_[{valid_scores[i]},{valid_scores[i+1]})"
                            new_df[new_col] = \
                            [1 if (x >= valid_scores[i] and x < valid_scores[i+1]) else 0 for x in new_df[col]]
                            i += 1
                        new_df = new_df.drop(columns=[col])
                    else:
                        # numerical column, add mask feature then proceed with Binning bins
                        new_df[f'{col}_mask'] = [
                            1 if pd.isna(x) else 0 for x in new_df[col]
                        ]

                        # get non-missing feature values acros whole dataframe
                        # --> compute bin boundaries and save in a dictionary
                        quantile_values = quantiles_dict[col]
                        if len(quantile_values) != len(np.unique(quantile_values)):
                            print(f"{col} has duplicate bins")
                            break

                        # save first bin starting from -inf (just to be safe)
                        new_df[f'{col}_[{-np.inf}, {quantile_values[0]})'] = \
                        [quantile_class(x, -np.inf, quantile_values[0]) for x in new_df[col]]

                        # use for loop to encode all the inner bins
                        num_bins = len(quantile_values) + 1
                        if num_bins > 2:
                            for i in range(0, num_bins - 2):
                                new_df[f'{col}_[{quantile_values[i]}, {quantile_values[i+1]})'] = \
                                [quantile_class(x, quantile_values[i], quantile_values[i+1]) for x in new_df[col]]

                        # extend last bin to inf (just to be safe)
                        new_df[f'{col}_[{quantile_values[-1]}, {np.inf}]'] = \
                        [quantile_class(x, quantile_values[-1], np.inf) for x in new_df[col]]

                        # remove original numerical feature column
                        new_df = new_df.drop(columns=[col])

            if not np.all(binning_df.columns == new_df.columns):
                print("columns don't match!")
            return new_df

        train_data_saved = train_data.copy()
        train_data, quantiles_dict = encode_train(train_data_saved, 4)
        val_data = encode_test(train_data_saved, val_data, quantiles_dict,
                            train_data)
        test_data = encode_test(train_data_saved, test_data, quantiles_dict,
                                train_data)

        return train_data, val_data, test_data

    @staticmethod
    def impute_data(train_data, val_data, test_data, method="mean"):
        if method == "mean":
            for col in train_data.columns:
                if col not in Imputer.static_cols:
                    mean_val = train_data[col].mean()
                    train_data[col].fillna(mean_val, inplace=True)
                    val_data[col].fillna(mean_val, inplace=True)
                    test_data[col].fillna(mean_val, inplace=True)
        elif method == "interpolate":
            train_data = Imputer.complete_imputation(train_data)
            val_data = Imputer.complete_imputation(val_data)
            test_data = Imputer.complete_imputation(test_data)
        elif method == "binning":
            train_data, val_data, test_data = Imputer.binning_imputation(train_data, val_data, test_data)
        return train_data, val_data, test_data


