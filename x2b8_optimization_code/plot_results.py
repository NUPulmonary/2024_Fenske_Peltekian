import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df = pd.read_csv("Final_Everything.csv").sort_values(by='AUC', ascending=False)

def parse_metric_value(metric_str):
    return float(metric_str.split(' ')[0])

def parse_confidence_interval(metric_str):
    ci_str = metric_str[metric_str.find("(")+1:metric_str.find(")")]
    ci = [float(x) for x in ci_str.split(',')]
    return ci

for metric in ['AUC', 'AUPRC', 'Overall Accuracy', 'F1 Score Class 0', 'F1 Score Class 1', 
               'Precision Class 0', 'Precision Class 1', 'Recall Class 0', 'Recall Class 1']:
    df[f'{metric}_value'] = df[metric].apply(parse_metric_value)
    df[[f'{metric}_ci_lower', f'{metric}_ci_upper']] = pd.DataFrame(df[metric].apply(parse_confidence_interval).tolist(), index=df.index)

sns.set_style("whitegrid")

metrics = ['AUC', 'AUPRC', 'Overall Accuracy', 'F1 Score Class 0', 'F1 Score Class 1', 
           'Precision Class 0', 'Precision Class 1', 'Recall Class 0', 'Recall Class 1']

for metric in metrics:
    f, ax = plt.subplots(figsize=(12, 6))

    df_long = pd.melt(df, id_vars=['Model_Type', 'Imputation_Method'], value_vars=[metric])

    df_long['metric_value'] = df_long['value'].str.extract(r'(\d+\.\d+)').astype(float)
    df_long['ci_lower'] = df_long['value'].str.extract(r'\((\d+\.\d+)').astype(float)
    df_long['ci_upper'] = df_long['value'].str.extract(r'(\d+\.\d+)\)').astype(float)
    palette = sns.color_palette("hsv", len(df['Model_Type'].unique()))
    sns.boxplot(x='Imputation_Method', y='metric_value', hue='Model_Type', data=df_long, ax=ax, dodge=True, palette=palette)

    for i, imputation_method in enumerate(df['Imputation_Method'].unique()):
        for j, model_type in enumerate(df['Model_Type'].unique()):
            color = palette[j]
            
            subset = df_long[(df_long['Model_Type'] == model_type) & (df_long['Imputation_Method'] == imputation_method)]

            if subset.empty:
                continue

            dodge_width = 0.8 / len(df['Model_Type'].unique())
            position = i + dodge_width * (j - (len(df['Model_Type'].unique()) - 1) / 2)

            ax.errorbar(
                position, 
                subset['metric_value'].mean(), 
                yerr=[[subset['metric_value'].mean() - subset['ci_lower'].mean()], 
                      [subset['ci_upper'].mean() - subset['metric_value'].mean()]],
                fmt='none', 
                ecolor=color,
                capsize=5,
                elinewidth=2
            )

    ax.set(xlabel="Imputation Method", ylabel=f"{metric} Value")
    sns.despine(left=True, bottom=True)

    plt.legend(title='Model Type')
    plt.title(f"{metric} Boxplot")
    plt.show()



import pandas as pd
import matplotlib.pyplot as plt

df_scores = pd.read_csv("fi_scores.csv")

def extract_feature_score(series):
    feature_score_split = series.str.rsplit(' ', n=1, expand=True)
    feature_score_split.columns = ['Feature', 'Score']
    feature_score_split['Score'] = pd.to_numeric(feature_score_split['Score'], errors='coerce')
    return feature_score_split


df1 = extract_feature_score(df_scores["Feature Importance Scores:"])
df2 = extract_feature_score(df_scores["Feature Importance Scores:.1"])
df3 = extract_feature_score(df_scores["Feature Importance Scores:.2"])
df4 = extract_feature_score(df_scores["Feature Importance Scores:.3"])

def process_dataframe(df):
    df = df.iloc[1:].copy()
    df['Feature'] = df['Feature'].str.strip()
    df['Feature'] = df['Feature'].str.split(' ').apply(lambda x: ' '.join(x[1:]))
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    return df


df1 = process_dataframe(df1)
df2 = process_dataframe(df2)
df3 = process_dataframe(df3)
df4 = process_dataframe(df4)
df1['Feature'] = df1['Feature'].str.replace(r'\s+', ' ', regex=True) 
df1['Feature'] = df1['Feature'].str.replace(r'[^\w\s]', '', regex=True)

df2['Feature'] = df2['Feature'].str.replace(r'\s+', ' ', regex=True) 
df2['Feature'] = df2['Feature'].str.replace(r'[^\w\s]', '', regex=True)

df3['Feature'] = df3['Feature'].str.replace(r'\s+', ' ', regex=True)
df3['Feature'] = df3['Feature'].str.replace(r'[^\w\s]', '', regex=True)

df4['Feature'] = df4['Feature'].str.replace(r'\s+', ' ', regex=True)
df4['Feature'] = df4['Feature'].str.replace(r'[^\w\s]', '', regex=True)

consolidated_df = pd.merge(df1, df2, on='Feature', how='outer', suffixes=['_1', '_2'])
consolidated_df = pd.merge(consolidated_df, df3, on='Feature', how='outer')
consolidated_df = pd.merge(consolidated_df, df4, on='Feature', how='outer', suffixes=['_3', '_4'])