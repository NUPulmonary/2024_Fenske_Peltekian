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
from keras.models import Model
from keras.layers import Input, Masking, SimpleRNN, Dropout, Dense, LSTM
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
    def train_and_evaluate(model, model_type, X_train, y_train, X_val, y_val, X_test, y_test, original_lengths_test, feature_names, train_data, val_data, test_data, hyperparams, print_args, is_deep_learning=True):               
        ModelTrainer.set_seed(ModelTrainer.SEED)
        # X_test = X_test_original.copy()
        base_feature_names = ['ECMO_flag', 'Hemodialysis_flag', 'CRRT_flag', 'Temperature',
             'Heart_rate', 'Systolic_blood_pressure', 'Diastolic_blood_pressure', 'Norepinephrine_rate',
             'Respiratory_rate', 'Oxygen_saturation', 'GCS_eye_opening', 'GCS_motor_response',
             'GCS_verbal_response', 'RASS_score', 'PEEP', 'FiO2', 'Plateau_Pressure', 'Lung_Compliance',
             'ABG_pH', 'ABG_PaCO2', 'ABG_PaO2', 'WBC_count', 'Lymphocytes', 'Neutrophils', 'Hemoglobin',
             'Platelets', 'Bicarbonate', 'Creatinine', 'Albumin', 'Bilirubin', 'CRP', 'D_dimer', 'Ferritin',
             'LDH', 'Lactic_acid', 'Procalcitonin','Minute_Ventilation']
        
        if is_deep_learning:
            ModelTrainer.set_seed(ModelTrainer.SEED)
            tf.config.run_functions_eagerly(True)
            callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min",
                restore_best_weights=True
            )
            # learning_rate = 0.01
            # optimizer = Adam(learning_rate=learning_rate)
            # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            if not print_args['training']:
                verbose = 0
            else:
                verbose = 'auto'
            history = model.fit(X_train,
                        y_train,
                        epochs=hyperparams['epochs'],
                        batch_size=hyperparams['batch_size'],
                        validation_data=(X_val, y_val),
                        callbacks=[callback],
                        verbose=verbose)
            if print_args['model_architecture']:
                print(model.summary())
            raw_preds = model.predict(X_test, verbose = verbose)
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

            base_feature_names = ['ECMO_flag', 'Intubation_flag', 'Hemodialysis_flag', 'CRRT_flag', 'Temperature',
             'Heart_rate', 'Systolic_blood_pressure', 'Diastolic_blood_pressure', 'Norepinephrine_rate',
             'Norepinephrine_flag', 'Respiratory_rate', 'Oxygen_saturation', 'GCS_eye_opening', 'GCS_motor_response',
             'GCS_verbal_response', 'RASS_score', 'PEEP', 'FiO2', 'Plateau_Pressure', 'Lung_Compliance',
             'ABG_pH', 'ABG_PaCO2', 'ABG_PaO2', 'WBC_count', 'Lymphocytes', 'Neutrophils', 'Hemoglobin',
             'Platelets', 'Bicarbonate', 'Creatinine', 'Albumin', 'Bilirubin', 'CRP', 'D_dimer', 'Ferritin',
             'LDH', 'Lactic_acid', 'Procalcitonin']
            y_test_flattened = [y_test[i, -original_lengths_test[i]:, 0] for i in range(len(y_test))]
            y_test_flattened = np.concatenate(y_test_flattened)

            preds_flattened = []
            start_idx = 0
            for length in original_lengths_test:
                end_idx = start_idx + length
                preds_flattened.extend(preds[start_idx:end_idx])
                start_idx = end_idx
            preds_flattened = np.array(preds_flattened)

            auc_base = roc_auc_score(y_test_flattened, preds_flattened)
            
            if print_args['run_feature_importance']:

                feature_importances = {}

                for base_feature in base_feature_names:
                    if print_args['feature_importance_masking']:
                        print(f"\nProcessing feature: {base_feature}")
                    X_test_modified = X_test.copy()
                    feature_indices = [i for i, name in enumerate(feature_names) if base_feature in name and not name.endswith("_mask")]
                    mask_index = feature_names.index(f"{base_feature}_mask") if f"{base_feature}_mask" in feature_names else None
                    if print_args['feature_importance_masking']:
                        print("Before modification:")
                    for patient_idx, length in enumerate(original_lengths_test[:1]):
                        if print_args['feature_importance_masking']:
                            print(f"Patient {patient_idx+1}, Original Length: {length}")
                        start_time_step = max(0, X_test_modified.shape[1] - length)
                        for time_step in range(start_time_step, X_test_modified.shape[1]):
                            if print_args['feature_importance_masking']:
                                print(f"Time Step {time_step+1-start_time_step}: Feature values {X_test_modified[patient_idx, time_step, feature_indices]}, Mask: {X_test_modified[patient_idx, time_step, mask_index] if mask_index is not None else 'N/A'}")

                    for patient_idx, length in enumerate(original_lengths_test):
                        if mask_index is not None:
                            X_test_modified[patient_idx, -length:, feature_indices] = 0
                            X_test_modified[patient_idx, -length:, mask_index] = 1
                    if print_args['feature_importance_masking']:                
                        print("After modification:")
                    for patient_idx, length in enumerate(original_lengths_test[:1]):
                        if print_args['feature_importance_masking']:
                            print(f"Patient {patient_idx+1}, Original Length: {length}")
                        start_time_step = max(0, X_test_modified.shape[1] - length)
                        for time_step in range(start_time_step, X_test_modified.shape[1]):
                            feature_values = X_test_modified[patient_idx, time_step, feature_indices]
                            mask_value = X_test_modified[patient_idx, time_step, mask_index] if mask_index is not None else 'N/A'
                            if print_args['feature_importance_masking']:
                                print(f"Time Step {time_step+1-start_time_step}: Feature values: {feature_values}, Mask: {mask_value}")
                
                    preds_shuffled = model.predict(X_test_modified, verbose = verbose)
                    preds_shuffled_flattened = []
                    for i in range(len(preds_shuffled)):
                        original_length = original_lengths_test[i]
                        patient_preds = preds_shuffled[i, -original_length:].ravel()
                        preds_shuffled_flattened.extend(patient_preds)
                    preds_shuffled_flattened = np.array(preds_shuffled_flattened)
                
                    auc_shuffled = roc_auc_score(y_test_flattened, preds_shuffled_flattened)
                
                    feature_importance = auc_base - auc_shuffled
                    feature_importances[base_feature] = feature_importance
                    if print_args['feature_importance_masking']:
                        print(f"Feature Importance for {base_feature}: {feature_importance}")

                feature_importance_df = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
                feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
                feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

                if print_args['feature_importance_masking']:
                    print("\nTop 10 Feature Importance Scores:")
                    print(feature_importance_df.head(10))

                top_feature_importance_df = feature_importance_df.head(10)

                plt.figure(figsize=(10, 5))
                sns.barplot(x='Importance', y='Feature', data=top_feature_importance_df, orient='h')
                plt.xlabel('Feature Importance Score')
                plt.ylabel('Feature')
                plt.title(f'{model_type} - Top 10 Feature Importance Scores')
                plt.tight_layout()
                plt.show()
            
            metric_results, _ = ModelTrainer.cd_metrics(y_true, y_preds_proba, X_test, threshold)
            print(f"AUC for {model_type}: {metric_results['AUC']}")
            return metric_results, y_true, np.ravel(preds), X_train, X_val, X_test, model
        
        else:
            model.fit(X_train, y_train)
            calibrated_clf = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
            calibrated_clf.fit(X_val, y_val)

            baseline_preds = calibrated_clf.predict_proba(X_test)[:, 1]
            baseline_auc = roc_auc_score(y_test, baseline_preds)
            counts = train_data['labels'].value_counts()
            threshold = 1 - counts[0] / (counts[0] + counts[1])
            print("Threshold", threshold)
            
            if print_args['run_feature_importance']:

                aggregated_feature_importances = {}

                for base_feature in base_feature_names:
                    valid_feature_indices = [idx for idx, feature in enumerate(feature_names) if base_feature in feature]
                    if not valid_feature_indices:
                        continue

                    X_test_modified = np.array(X_test, copy=True)

                    for idx in valid_feature_indices:
                        if idx < X_test_modified.shape[1]:
                            X_test_modified[:, idx] = 0

                    modified_preds = calibrated_clf.predict_proba(X_test_modified)[:, 1]
                    modified_auc = roc_auc_score(y_test, modified_preds)

                    aggregated_feature_importances[base_feature] = baseline_auc - modified_auc
            
            

                feature_importance_df = pd.DataFrame(list(aggregated_feature_importances.items()), columns=['Feature', 'Importance'])
                feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
            
                if print_args['feature_importance_masking']:
                    print("\nTop 10 Feature Importance Scores:")
                    print(feature_importance_df.head(10))

                top_feature_importance_df = feature_importance_df.head(10)

                plt.figure(figsize=(10, 5))
                sns.barplot(x='Importance', y='Feature', data=top_feature_importance_df, orient='h')
                plt.xlabel('Feature Importance Score')
                plt.ylabel('Feature')
                plt.title(f'{model_type} - Top 10 Feature Importance Scores')
                plt.tight_layout()
                plt.show()
                
                
                

           
            if model_type in ['XGBoost','LightGBM']:
               if print_args['save_shap']:
                   explainer = shap.Explainer(model)
                   shap_values = explainer.shap_values(X_test)
                   shap.summary_plot(shap_values, X_test, feature_names=feature_names,max_display=10,show=False)
                   # plt.savefig(f'shap_{model_type}.png')
                   plt.savefig(f'shap_{model_type}.pdf', format='pdf', bbox_inches='tight')
                   plt.show()
            # metric_results, _ = ModelTrainer.cd_metrics(y_test, baseline_preds, X_test, threshold)
            # print(f"AUC for {model_type}: {metric_results['AUC']}")
            # return metric_results, y_test, baseline_preds, model
            metric_results, _ = ModelTrainer.cd_metrics(y_test, baseline_preds, X_test, threshold)
            print(f"AUC for {model_type}: {metric_results['AUC']}")
            return metric_results, y_test, baseline_preds, X_train, X_val, X_test, model

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
        np.random.seed(42)
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

        # cm = confusion_matrix(y_true, y_preds)
        # cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
        # cm_display.plot(cmap=plt.cm.Blues)
        # plt.show()
        
        return metric_results, results

    # @staticmethod
    # def build_dl_model(model_type, input_shape):
    #     inputs = Input(shape=input_shape)
    #     masked_inputs = Masking(mask_value=0.0)(inputs)

    #     if model_type == 'RNN':
    #         RNN_1 = SimpleRNN(64, return_sequences=True, activation='sigmoid')(masked_inputs)
    #         dropout_1 = Dropout(0.2)(RNN_1)
    #         RNN_2 = SimpleRNN(64, return_sequences=True, activation='sigmoid')(dropout_1)
    #         dropout_2 = Dropout(0.2)(RNN_2)
    #         outputs = Dense(1, activation='sigmoid')(dropout_2)
    #     elif model_type == 'LSTM':
    #         LSTM_1 = LSTM(32, return_sequences=True, activation='sigmoid')(masked_inputs)
    #         dropout_1 = Dropout(0.2)(LSTM_1)
    #         LSTM_2 = LSTM(32, return_sequences=True, activation='sigmoid')(dropout_1)
    #         dropout_2 = Dropout(0.2)(LSTM_2)
    #         outputs = Dense(1, activation='sigmoid')(dropout_2)

    #     model = Model(inputs=inputs, outputs=outputs)

    #     learning_rate=0.01
    #     optimizer = Adam(learning_rate=learning_rate)
    #     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    #     return model
    @staticmethod
    def create_dl_model(model_type, input_shape, hyperparams):
        inputs = Input(shape=input_shape)
        masked_inputs = Masking(mask_value=0.0)(inputs)

        if model_type == 'RNN':
            RNN_1 = SimpleRNN(hyperparams['layer_1_size'], return_sequences=True, activation='sigmoid')(masked_inputs)
            dropout_1 = Dropout(0.2)(RNN_1)
            RNN_2 = SimpleRNN(hyperparams['layer_2_size'], return_sequences=True, activation='sigmoid')(dropout_1)
            dropout_2 = Dropout(0.2)(RNN_2)
            outputs = Dense(1, activation='sigmoid')(dropout_2)
        elif model_type == 'LSTM':
            LSTM_1 = LSTM(hyperparams['layer_1_size'], return_sequences=True, activation='sigmoid')(masked_inputs)
            dropout_1 = Dropout(0.2)(LSTM_1)
            LSTM_2 = LSTM(hyperparams['layer_2_size'], return_sequences=True, activation='sigmoid')(dropout_1)
            dropout_2 = Dropout(0.2)(LSTM_2)
            outputs = Dense(1, activation='sigmoid')(dropout_2)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model = Model(inputs=inputs, outputs=outputs)

        optimizer = Adam(learning_rate=hyperparams['learning_rate'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    @staticmethod
    
    def run_model(train_data, val_data, test_data, model_type, imputation_method, hyperparams, print_args):
        train_data, val_data, test_data = Imputer.impute_data(train_data, val_data, test_data, imputation_method)
        hyperparams = hyperparams[f'{model_type}_{imputation_method}']
        if model_type in ['RNN', 'LSTM']:
            X_train_len = train_data.Patient_id_2.value_counts().values[0] + 25
            X_train, y_train, original_lengths_train, feature_names = Preprocessor.preprocess_data_for_sequences(train_data, X_train_len)
            X_val, y_val, original_lengths_val, feature_names = Preprocessor.preprocess_data_for_sequences(val_data, X_train_len)
            X_test, y_test, original_lengths_test, feature_names = Preprocessor.preprocess_data_for_sequences(test_data, X_train_len)
            input_shape = (X_train_len, X_train.shape[2])
            model = ModelTrainer.create_dl_model(model_type, input_shape, hyperparams)
            metric_results, y_true, preds_proba, X_train, X_val, X_test, model = ModelTrainer.train_and_evaluate(model, model_type, X_train, y_train, X_val, y_val, X_test, y_test, original_lengths_test, feature_names, train_data, val_data, test_data, hyperparams, print_args, is_deep_learning=True)
        
        else:
            X_train, y_train, X_val, y_val, X_test, y_test, feature_names = Preprocessor.preprocess_data(train_data, val_data, test_data, model_type)
            if model_type == 'XGBoost':
                model = xgboost.XGBClassifier(n_iterations=1000, max_depth=30)
            elif model_type == 'XGBoostMinusGCSVerbal':
                model = xgboost.XGBClassifier(random_state=ModelTrainer.SEED)
            elif model_type == 'XGBoostTopTen':
                model = xgboost.XGBClassifier(random_state=ModelTrainer.SEED)
            elif model_type == 'LightGBM':
                hyperparams['verbose'] = -1
                model = lgb.LGBMClassifier(**hyperparams, random_state=ModelTrainer.SEED)
            elif model_type == 'RandomForest':
                model = RandomForestClassifier(random_state=ModelTrainer.SEED)
            elif model_type == 'LogisticRegression':
                model = LogisticRegression(**hyperparams, random_state=ModelTrainer.SEED)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            metric_results, y_true, preds_proba, X_train, X_val, X_test, model = ModelTrainer.train_and_evaluate(model, model_type, X_train, y_train, X_val, y_val, X_test, y_test, None, feature_names, train_data, val_data, test_data, hyperparams, print_args, is_deep_learning=False)
        return metric_results, y_true, preds_proba, X_train, X_val, X_test, model, model_type