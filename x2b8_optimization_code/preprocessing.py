import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings("ignore")

class Preprocessor:
    @staticmethod
    def preprocess_data(train_data, val_data, test_data, model_type):
        common_drop_cols = [
            'stay', 'day', 'Patient_id', 'Patient_id_2', 'old_extubation_status',
            'extubation_status', 'patient', 'patient_day', 'trach_collar_status',
            'next_day_extubation_status', 'Norepinephrine_flag', 'Intubation_flag'
        ]

        drop_cols = common_drop_cols.copy()

        if model_type == 'XGBoostTopTen':
            keep_cols = [
                'GCS_verbal_response', 'Plateau_Pressure', 'Heart_rate',
                'GCS_motor_response', 'Respiratory_rate', 'Oxygen_saturation',
                'PEEP', 'GCS_eye_opening', 'Systolic_blood_pressure',
                'Diastolic_blood_pressure', 'labels'
            ]
            drop_cols = [col for col in train_data.columns if col not in keep_cols]

        elif model_type == 'XGBoostMinusGCSVerbal':
            if 'GCS_verbal_response' in train_data.columns:
                drop_cols.append('GCS_verbal_response')

        train_data = train_data.drop(columns=drop_cols)
        val_data = val_data.drop(columns=drop_cols)
        test_data = test_data.drop(columns=drop_cols)

        X_train = train_data.drop('labels', axis=1).values
        X_val = val_data.drop('labels', axis=1).values
        X_test = test_data.drop('labels', axis=1).values

        y_train = list(train_data['labels'].values)
        y_val = list(val_data['labels'].values)
        y_test = list(test_data['labels'].values)

        feature_names = train_data.columns.tolist()
        # print(type(feature_names))
        # print(feature_names)

        return X_train, y_train, X_val, y_val, X_test, y_test, feature_names

    @staticmethod
    def preprocess_data_for_sequences(df, max_len):
        excluded_columns = [
            'labels', 'stay', 'day', 'Patient_id', 'Patient_id_2',
            'old_extubation_status', 'extubation_status', 'patient',
            'patient_day', 'trach_collar_status', 'next_day_extubation_status', 'Norepinephrine_flag', 'Intubation_flag'
        ]

        feature_names = [col for col in df.columns if col not in excluded_columns]
        X = [
            np.array(df[df.Patient_id_2 == patient].drop(columns=excluded_columns)) 
            for patient in df.Patient_id_2.unique()
        ]
        y = [
            np.array(df[df.Patient_id_2 == patient]['labels'])
            for patient in df.Patient_id_2.unique()
        ]


        original_lengths = [len(sequence) for sequence in X]

        X_padded = pad_sequences(X, maxlen=max_len, dtype='float32', padding='pre')
        y_padded = pad_sequences(y, maxlen=max_len, dtype='float32', padding='pre')
        y_padded = np.expand_dims(y_padded, axis=-1)

        return X_padded, y_padded, original_lengths, feature_names

