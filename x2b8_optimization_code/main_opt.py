import pandas as pd
from imputation import Imputer
from preprocessing import Preprocessor
from model_training_opt import ModelTrainer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings("ignore")
import matplotlib.font_manager as font_manager
from sklearn.metrics import roc_auc_score
import random
import os
import tensorflow as tf

def set_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    tf.random.set_seed(seed_value)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

SEED = 42
set_seed(SEED)

def load_data():
    train_data = pd.read_csv('../data/train.csv', index_col=0)
    print(train_data.shape)
    val_data = pd.read_csv('../data/val.csv', index_col=0)
    print(val_data.shape)
    test_data = pd.read_csv('../data/test.csv', index_col=0)
    print(test_data.shape)
    return train_data, val_data, test_data

def run_models_and_collect_results(model_types, imputation_methods):
    all_results = []
    prediction_data = {}
    best_hyperparameters = {}
    valid_models = ['XGBoost', 'XGBoostMinusGCSVerbal', 'XGBoostTopTen', 'RNN', 'LSTM', 'LightGBM', 'RandomForest', 'LogisticRegression']
    valid_imputations = ['binning', 'mean', 'raw']
    
    for model_type in model_types:
        if model_type not in valid_models:
            print("Unrecognizable model:", model_type)
            continue

        for method in imputation_methods:
            if method not in valid_imputations:
                print("Unrecognizable imputation method:", method)
                continue

            if (model_type in ['RNN', 'LSTM', 'RandomForest', 'LogisticRegression'] and method == 'raw') or \
               (model_type in ['XGBoostMinusGCSVerbal', 'XGBoostTopTen'] and method in ['binning', 'mean']):
                continue

            train_data, val_data, test_data = load_data()
            print(f"Running for model: {model_type} with imputation method: {method}")

            tf.keras.backend.clear_session()
            set_seed(SEED)

            metric_results, y_true, preds_proba, best_params, X_train, X_val, X_test, model, model_type = ModelTrainer.run_model(train_data, val_data, test_data, model_type, method)
            
            result_dict = {
                'Model_Type': model_type,
                'Imputation_Method': method,
            }
            
            result_dict.update(metric_results)
            all_results.append(result_dict)

            prediction_key = f"{model_type}_{method}"
            prediction_data[prediction_key] = {'y_true': y_true, 'y_preds_proba': preds_proba}

            best_hyperparameters[prediction_key] = best_params

    return pd.DataFrame(all_results), prediction_data, best_hyperparameters

def plot_metrics(ax_roc, ax_pr, y_true, y_preds_proba, label, color):
    y_preds_proba = np.ravel(y_preds_proba)
    y_true = np.ravel(y_true)

    arial_font = font_manager.FontProperties(family='Arial', size=18)
    arial_font_legend = font_manager.FontProperties(family='Arial', size=12)
    fpr, tpr, _ = roc_curve(y_true, y_preds_proba)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {roc_auc:.4f})')

    y_true_minority = np.where(y_true == 1, 0, 1)
    y_preds_proba_minority = 1 - y_preds_proba
    precision_minority, recall_minority, _ = precision_recall_curve(y_true_minority, y_preds_proba_minority)
    average_precision_minority = average_precision_score(y_true_minority, y_preds_proba_minority)
    ax_pr.plot(recall_minority, precision_minority, color=color, lw=2, label=f'{label} Minority (AP = {average_precision_minority:.4f})')

    ax_roc.set_title('Receiver Operating Characteristic', fontproperties=arial_font)
    ax_roc.set_xlabel('False Positive Rate (1-Specificity)', fontproperties=arial_font)
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontproperties=arial_font)
    
    ax_pr.set_title('Precision-Recall Curve', fontproperties=arial_font)
    ax_pr.set_xlabel('Recall (Sensitivity)', fontproperties=arial_font)
    ax_pr.set_ylabel('Precision (Positive predictive value)', fontproperties=arial_font)
    
    ax_roc.legend(loc="lower right", prop=arial_font_legend)
    ax_pr.legend(loc="best", prop=arial_font_legend)
    return fpr, tpr, roc_auc

def plot_all_metrics(results_df, prediction_data):
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(20, 6))
    arial_font = font_manager.FontProperties(family='Arial', size=18)
    arial_font_legend = font_manager.FontProperties(family='Arial', size=12)
    color_cycle = ['#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B',
                     '#ECAE8A', '#FF9DA6', '#9D755D', '#BAB0AC', '#9C755F']

    lines_roc, lines_pr = [], []

    for i, (key, data) in enumerate(prediction_data.items()):
        y_true = data['y_true']
        y_preds_proba = data['y_preds_proba']
        label = key.replace('_', ' + ')
        color = color_cycle[i % len(color_cycle)]
        print(f"Plotting {label} with color {color}")
        unique, counts = np.unique(y_preds_proba, return_counts=True)
        print("Unique predicted probabilities and their counts:")
        print(np.asarray((unique, counts)).T)
        y_true = np.array(y_true)
        y_true_minority = np.where(y_true == 0, 1, 0)
        y_preds_proba_minority = 1 - y_preds_proba

        plot_metrics(ax_roc, ax_pr, y_true, y_preds_proba, label, color)

        line_roc = ax_roc.get_lines()[-1]
        auc_value = roc_auc_score(y_true, y_preds_proba)
        lines_roc.append((line_roc, f'{label} (AUC = {auc_value:.4f})'))

        ap_value_minority = average_precision_score(y_true_minority, y_preds_proba_minority)
        lines_pr.append((ax_pr.get_lines()[-1], f'{label} (AP = {ap_value_minority:.4f})'))
        fpr, tpr, roc_auc = plot_metrics(ax_roc, ax_pr, y_true, y_preds_proba, label, color)
        
        ax_roc.set_xlim([min(fpr) - 0.05, max(fpr) + 0.05])
        ax_roc.set_ylim([min(tpr) - 0.05, max(tpr) + 0.05])

        ax_pr.set_xlim([-0.05, 1.05])
        ax_pr.set_ylim([-0.05, 1.05])

    _, _, test_data = load_data()

    minority_class_proportion = test_data['labels'].value_counts(normalize=True).get(0, 1)
    print("minority_class_proportion", minority_class_proportion)

    baseline_roc_line, = ax_roc.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', label='Baseline (data balance - no skill)')
    baseline_pr_line = ax_pr.axhline(y=minority_class_proportion, color='black', lw=1, linestyle='--', label='Baseline (minority class proportion)')

    lines_roc.append((baseline_roc_line, 'Baseline (data balance - no skill)'))
    lines_roc.sort(key=lambda x: (x[1].lower() == 'baseline (data balance - no skill)', -float(x[1].split('= ')[-1].rstrip(')')) if 'auc' in x[1].lower() else 0))
    sorted_lines_roc, sorted_labels_roc = zip(*lines_roc)

    lines_pr.sort(key=lambda x: float(x[1].split('= ')[-1].rstrip(')')), reverse=True)
    lines_pr.append((baseline_pr_line, 'Baseline (data balance - no skill)'))
    sorted_lines_pr, sorted_labels_pr = zip(*lines_pr)

    ax_roc.set_title('Receiver Operating Characteristic', fontproperties=arial_font)
    ax_roc.set_xlabel('False Positive Rate (1-Specificity)', fontproperties=arial_font)
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontproperties=arial_font)

    ax_pr.set_title('Precision-Recall Curve', fontproperties=arial_font)
    ax_pr.set_xlabel('Recall (Sensitivity)', fontproperties=arial_font)
    ax_pr.set_ylabel('Precision (Positive predictive value)', fontproperties=arial_font)

    ax_roc.legend(sorted_lines_roc, sorted_labels_roc, loc="lower right", prop=arial_font_legend)
    ax_pr.legend(sorted_lines_pr, sorted_labels_pr, loc="best", prop=arial_font_legend)

    plt.show()

def main():
    model_types = ['LSTM']
    imputation_methods = ['binning']
    #['XGBoost', 'XGBoostMinusGCSVerbal', 'XGBoostTopTen', 'RNN', 'LSTM', 'LightGBM', 'RandomForest', 'LogisticRegression']
    #['binning', 'mean', 'raw']

    train_data, val_data, test_data = load_data()
    results_df, prediction_data, best_hyperparameters = run_models_and_collect_results(model_types, imputation_methods)

    results_df.to_csv('void.csv', index=False)
    plot_all_metrics(results_df, prediction_data)

    for key, params in best_hyperparameters.items():
        print(f"Best hyperparameters for {key}: {params}")

if __name__ == "__main__":
    main()
