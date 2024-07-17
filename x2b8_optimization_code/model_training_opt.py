import numpy as np
import pandas as pd
import xgboost
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, ConfusionMatrixDisplay)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking, SimpleRNN, Dropout, Dense, LSTM
import matplotlib.pyplot as plt
import shap
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

from imputation import Imputer 
from preprocessing import Preprocessor
from scipy.stats import mode
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.optimizers import Adam
import random
import os
import warnings
import sys
warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from itertools import product

class ModelTrainer:
    @staticmethod
    def set_seed(seed_value):
        np.random.seed(seed_value)
        random.seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        tf.random.set_seed(seed_value)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        os.environ["CUDA_VISIBLE_DEVICES"] = "" 

    SEED = 42
    set_seed(SEED)

    @staticmethod
    def create_dl_model(model_type, input_shape, learning_rate=0.01, units_rnn_1=128, units_rnn_2=64):
        inputs = Input(shape=input_shape)
        masked_inputs = Masking(mask_value=0.0)(inputs)
        ModelTrainer.set_seed(ModelTrainer.SEED)
        if model_type == 'RNN':
            RNN_1 = SimpleRNN(units_rnn_1, return_sequences=True, activation='sigmoid')(masked_inputs)
            dropout_1 = Dropout(0.2)(RNN_1)
            RNN_2 = SimpleRNN(units_rnn_2, return_sequences=True, activation='sigmoid')(dropout_1)
            dropout_2 = Dropout(0.2)(RNN_2)
            outputs = Dense(1, activation='sigmoid')(dropout_2)
        elif model_type == 'LSTM':
            LSTM_1 = LSTM(units_rnn_1, return_sequences=True, activation='sigmoid')(masked_inputs)
            dropout_1 = Dropout(0.2)(LSTM_1)
            LSTM_2 = LSTM(units_rnn_2, return_sequences=True, activation='sigmoid')(dropout_1)
            dropout_2 = Dropout(0.2)(LSTM_2)
            outputs = Dense(1, activation='sigmoid')(dropout_2)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model = Model(inputs=inputs, outputs=outputs)

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def grid_search_dl(model_type, X_train, y_train, X_val, y_val, original_lengths_val, val_data):
        learning_rates = [0.001, 0.01, 0.1]
        units_rnn_1_options = [32, 64, 128]
        units_rnn_2_options = [16, 32, 64]
        batch_sizes = [32, 64, 128]
        # learning_rates = [0.01]
        # units_rnn_1_options = [128]
        # units_rnn_2_options = [64]
        # batch_sizes = [32]
        best_auc = 0
        best_params = None
        best_model = None

        input_shape = (X_train.shape[1], X_train.shape[2])

        for lr, units1, units2, batch_size in product(learning_rates, units_rnn_1_options, units_rnn_2_options, batch_sizes):
            ModelTrainer.set_seed(ModelTrainer.SEED)
            model = ModelTrainer.create_dl_model(model_type, input_shape, learning_rate=lr, units_rnn_1=units1, units_rnn_2=units2)
            callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min",
                restore_best_weights=True
            )
            ModelTrainer.set_seed(ModelTrainer.SEED)
            history = model.fit(X_train, y_train,
                                epochs=100,
                                batch_size=batch_size,
                                validation_data=(X_val, y_val),
                                callbacks=[callback])
                                
            raw_preds = model.predict(X_val)
            mask_val = raw_preds[0][0][0]
            preds = []
            for patient in raw_preds:
                [preds.append(x) for x in patient if x != mask_val]

            y_true = val_data['labels']
            y_preds_proba = preds
            preds = np.ravel(preds)

            X_val = X_val.copy()
            y_val_flattened = [y_val[i, -original_lengths_val[i]:, 0] for i in range(len(y_val))]
            y_val_flattened = np.concatenate(y_val_flattened)

            preds_flattened = []
            start_idx = 0
            for length in original_lengths_val:
                end_idx = start_idx + length
                preds_flattened.extend(preds[start_idx:end_idx])
                start_idx = end_idx
            preds_flattened = np.array(preds_flattened)

            auc = roc_auc_score(y_val_flattened, preds_flattened)
            print(auc)
            if auc > best_auc:
                best_auc = auc
                best_params = (lr, units1, units2, batch_size)
                best_model = model

        print(f"Best params for {model_type}: {best_params}, Best AUC: {best_auc}")
        return best_model, best_params

    @staticmethod
    def optimize_hyperparameters_lr(X_train, y_train, X_val, y_val):
        logistic_params = {
            'C': [0.01, 0.1, 1],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
            # 'C': [0.01],
            # 'penalty': ['l1', 'l2'],
            # 'solver': ['liblinear']
        }

        X_combined = np.concatenate((X_train, X_val), axis=0)
        y_combined = np.concatenate((y_train, y_val), axis=0)

        test_fold = np.concatenate((-1 * np.ones(len(X_train)), np.zeros(len(X_val))))
        ps = PredefinedSplit(test_fold=test_fold)

        logistic_model = LogisticRegression(random_state=ModelTrainer.SEED)
        grid_search_lr = GridSearchCV(estimator=logistic_model, param_grid=logistic_params, cv=ps, scoring='roc_auc')
        grid_search_lr.fit(X_combined, y_combined)

        return grid_search_lr.best_params_

    @staticmethod
    def optimize_hyperparameters_lgb(X_train, y_train, X_val, y_val):
        lgb_params = {
            'num_leaves': [31, 61, 127],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 500]
        }
        # lgb_params = {
        #     'num_leaves': [31],
        #     'learning_rate': [0.01, 0.05],
        #     'n_estimators': [100]
        # }

        X_combined = np.concatenate((X_train, X_val), axis=0)
        y_combined = np.concatenate((y_train, y_val), axis=0)

        test_fold = np.concatenate((-1 * np.ones(len(X_train)), np.zeros(len(X_val))))
        ps = PredefinedSplit(test_fold=test_fold)

        lgb_model = lgb.LGBMClassifier(random_state=ModelTrainer.SEED)
        grid_search_lgb = GridSearchCV(estimator=lgb_model, param_grid=lgb_params, cv=ps, scoring='roc_auc')
        grid_search_lgb.fit(X_combined, y_combined)

        means = grid_search_lgb.cv_results_['mean_test_score']
        stds = grid_search_lgb.cv_results_['std_test_score']
        params = grid_search_lgb.cv_results_['params']
        
        for mean, std, param in zip(means, stds, params):
            print(f"Validation AUC: {mean:.4f} (+/-{std * 2:.4f}) for {param}")

        return grid_search_lgb.best_params_
    @staticmethod
    def train_and_evaluate(model, model_type, X_train, y_train, X_val, y_val, X_test, y_test, original_lengths_val, original_lengths_test, feature_names, train_data, val_data, test_data, is_deep_learning=True, epochs=100, batch_size=32, imputation_method="raw"):
        base_feature_names = ['ECMO_flag', 'Hemodialysis_flag', 'CRRT_flag', 'Temperature',
            'Heart_rate', 'Systolic_blood_pressure', 'Diastolic_blood_pressure', 'Norepinephrine_rate',
            'Respiratory_rate', 'Oxygen_saturation', 'GCS_eye_opening', 'GCS_motor_response',
            'GCS_verbal_response', 'RASS_score', 'PEEP', 'FiO2', 'Plateau_Pressure', 'Lung_Compliance',
            'ABG_pH', 'ABG_PaCO2', 'ABG_PaO2', 'WBC_count', 'Lymphocytes', 'Neutrophils', 'Hemoglobin',
            'Platelets', 'Bicarbonate', 'Creatinine', 'Albumin', 'Bilirubin', 'CRP', 'D_dimer', 'Ferritin',
            'LDH', 'Lactic_acid', 'Procalcitonin']
        ModelTrainer.set_seed(ModelTrainer.SEED)        
        if is_deep_learning:
            ModelTrainer.set_seed(ModelTrainer.SEED)
            tf.config.run_functions_eagerly(True)
            model, best_params = ModelTrainer.grid_search_dl(model_type, X_train, y_train, X_val, y_val, original_lengths_val, val_data)
            learning_rate, units_rnn_1, units_rnn_2, batch_size = best_params

            # callback = tf.keras.callbacks.EarlyStopping(
            #     monitor="val_loss",
            #     patience=5,
            #     mode="min",
            #     restore_best_weights=True
            # )
            # learning_rate = 0.01
            # optimizer = Adam(learning_rate=learning_rate)
            # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            # history = model.fit(X_train,
            #             y_train,
            #             epochs=epochs,
            #             batch_size=batch_size,
            #             validation_data=(X_val, y_val),
            #             callbacks=[callback])
            print(model.summary())            
            raw_preds = model.predict(X_test)
            mask_val = raw_preds[0][0][0]
            preds = []
            for patient in raw_preds:
                [preds.append(x) for x in patient if x != mask_val]

            counts = train_data['labels'].value_counts()
            threshold = 1 - counts[0.0] / (counts[0.0] + counts[1.0])
            print("Threshold", threshold)

            y_pred_classes = [1 if prob > threshold else 0 for prob in preds]

            y_true = test_data['labels']
            y_preds_proba = preds
            preds = np.ravel(preds)

            metric_results, _ = ModelTrainer.cd_metrics(y_true, y_preds_proba, X_test, threshold)
            print(f"AUC for {model_type}: {metric_results['AUC']}")
            return metric_results, y_true, np.ravel(preds), best_params, X_train, X_val, X_test, model, model_type
        
        else:
            if model_type == 'LogisticRegression':
                best_params = ModelTrainer.optimize_hyperparameters_lr(X_train, y_train, X_val, y_val)
                model = LogisticRegression(**best_params, random_state=ModelTrainer.SEED)
            elif model_type == 'LightGBM':
                best_params = ModelTrainer.optimize_hyperparameters_lgb(X_train, y_train, X_val, y_val)
                model = lgb.LGBMClassifier(**best_params, random_state=ModelTrainer.SEED)

            model.fit(X_train, y_train)
            calibrated_clf = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
            calibrated_clf.fit(X_val, y_val)

            baseline_preds = calibrated_clf.predict_proba(X_test)[:, 1]
            baseline_auc = roc_auc_score(y_test, baseline_preds)
            counts = train_data['labels'].value_counts()
            threshold = 1 - counts[0] / (counts[0] + counts[1])
            print("Threshold", threshold)

            metric_results, _ = ModelTrainer.cd_metrics(y_test, baseline_preds, X_test, threshold)
            print(f"AUC for {model_type}: {metric_results['AUC']}")
            return metric_results, y_test, baseline_preds, best_params, X_train, X_val, X_test, model, model_type


    @staticmethod
    def cd_metrics(y_true, y_preds_proba, X_test, threshold):
        y_true = np.array(y_true)
        y_preds_proba = np.array(y_preds_proba)
        y_preds = np.array([1 if x > threshold else 0 for x in y_preds_proba])

        n_iterations = 1000
        n_size = len(y_true)
        alpha = 0.95
        lower_p = ((1.0 - alpha) / 2.0) * 100
        upper_p = (alpha + ((1.0 - alpha) / 2.0)) * 100

        fpr, tpr, _ = roc_curve(y_true, y_preds_proba)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true, y_preds_proba)
        average_precision = average_precision_score(y_true, y_preds_proba)

        results = {
            "AUC": [],
            "AUPRC": [],
            "Overall Accuracy": [],
            "F1 Score Class 0": [],
            "F1 Score Class 1": [],
            "Precision Class 0": [],
            "Precision Class 1": [],
            "Recall Class 0": [],
            "Recall Class 1": []
        }

        for i in range(n_iterations):
            indices = np.random.choice(np.arange(n_size), size=n_size, replace=True)
            y_resampled = y_true[indices]
            y_preds_resampled = y_preds[indices]
            y_preds_proba_resampled = y_preds_proba[indices]

            if len(np.unique(y_resampled)) > 1:
                results["AUC"].append(roc_auc_score(y_resampled, y_preds_proba_resampled))
                y_resampled_inverted = 1 - y_resampled
                y_preds_proba_inverted = 1 - y_preds_proba_resampled
                results["AUPRC"].append(average_precision_score(y_resampled_inverted, y_preds_proba_inverted))

            results["Overall Accuracy"].append(accuracy_score(y_resampled, y_preds_resampled))

            f1_scores = f1_score(y_resampled, y_preds_resampled, average=None, zero_division=0)
            precision_scores = precision_score(y_resampled, y_preds_resampled, average=None, zero_division=0)
            recall_scores = recall_score(y_resampled, y_preds_resampled, average=None, zero_division=0)
            results["F1 Score Class 0"].append(f1_scores[0])
            results["F1 Score Class 1"].append(f1_scores[1])
            results["Precision Class 0"].append(precision_scores[0])
            results["Precision Class 1"].append(precision_scores[1])
            results["Recall Class 0"].append(recall_scores[0])
            results["Recall Class 1"].append(recall_scores[1])

        metric_results = {}
        for metric, scores in results.items():
            lower = np.percentile(scores, lower_p)
            upper = np.percentile(scores, upper_p)
            mean_score = np.mean(scores)
            formatted_result = f"{mean_score:.3f} ({lower:.3f}, {upper:.3f})"
            metric_results[metric] = formatted_result
        return metric_results, results

    # @staticmethod
    # def build_dl_model(model_type, input_shape):
    #     inputs = Input(shape=input_shape)
    #     masked_inputs = Masking(mask_value=0.0)(inputs)

    #     if model_type == 'RNN':
    #         RNN_1 = SimpleRNN(64, return_sequences=True, activation='sigmoid')(masked_inputs)
    #         dropout_1 = Dropout(0.2)(RNN_1)
    #         RNN_2 = SimpleRNN(32, return_sequences=True, activation='sigmoid')(dropout_1)
    #         dropout_2 = Dropout(0.2)(RNN_2)
    #         outputs = Dense(1, activation='sigmoid')(dropout_2)
    #     elif model_type == 'LSTM':
    #         LSTM_1 = LSTM(64, return_sequences=True, activation='sigmoid')(masked_inputs)
    #         dropout_1 = Dropout(0.2)(LSTM_1)
    #         LSTM_2 = LSTM(32, return_sequences=True, activation='sigmoid')(dropout_1)
    #         dropout_2 = Dropout(0.2)(LSTM_2)
    #         outputs = Dense(1, activation='sigmoid')(dropout_2)
    #     else:
    #         raise ValueError(f"Unknown model type: {model_type}")

    #     model = Model(inputs=inputs, outputs=outputs)
    #     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #     return model

    @staticmethod
    def run_model(train_data, val_data, test_data, model_type, imputation_method):
        train_data, val_data, test_data = Imputer.impute_data(train_data, val_data, test_data, imputation_method)
        if model_type in ['RNN', 'LSTM']:
            X_train_len = train_data.Patient_id_2.value_counts().values[0] + 25
            
            X_train, y_train, original_lengths_train, feature_names = Preprocessor.preprocess_data_for_sequences(train_data, X_train_len)
            X_val, y_val, original_lengths_val, feature_names = Preprocessor.preprocess_data_for_sequences(val_data, X_train_len)
            X_test, y_test, original_lengths_test, feature_names = Preprocessor.preprocess_data_for_sequences(test_data, X_train_len)
            input_shape = (X_train_len, X_train.shape[2])
            model = ModelTrainer.create_dl_model(model_type, input_shape, learning_rate=0.01, units_rnn_1=32, units_rnn_2=32)
            metric_results, y_true, preds_proba, best_params, X_train, X_val, X_test, model, model_type = ModelTrainer.train_and_evaluate(
                model, model_type, X_train, y_train, X_val, y_val, X_test, y_test, 
                original_lengths_val, original_lengths_test, feature_names, train_data, val_data, test_data, 
                is_deep_learning=True
            )
        
        else:
            X_train, y_train, X_val, y_val, X_test, y_test, feature_names = Preprocessor.preprocess_data(train_data, val_data, test_data, model_type)
            if model_type == 'XGBoost':
                model = xgboost.XGBClassifier(n_iterations=1000, max_depth=30)
            elif model_type == 'XGBoostMinusGCSVerbal':
                model = xgboost.XGBClassifier(random_state=ModelTrainer.SEED)
            elif model_type == 'XGBoostTopTen':
                model = xgboost.XGBClassifier(random_state=ModelTrainer.SEED)
            elif model_type == 'LightGBM':
                best_params = ModelTrainer.optimize_hyperparameters_lgb(X_train, y_train, X_val, y_val)
                model = lgb.LGBMClassifier(**best_params, random_state=ModelTrainer.SEED)
            elif model_type == 'RandomForest':
                model = RandomForestClassifier(random_state=ModelTrainer.SEED)
            elif model_type == 'LogisticRegression':
                best_params = ModelTrainer.optimize_hyperparameters_lr(X_train, y_train, X_val, y_val)
                model = LogisticRegression(**best_params, random_state=ModelTrainer.SEED)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            metric_results, y_true, preds_proba, best_params, X_train, X_val, X_test, model, model_type = ModelTrainer.train_and_evaluate(
                model, model_type, X_train, y_train, X_val, y_val, X_test, y_test, 
                None, None, feature_names, train_data, val_data, test_data, 
                is_deep_learning=False
            )
        return metric_results, y_true, preds_proba, best_params, X_train, X_val, X_test, model, model_type
