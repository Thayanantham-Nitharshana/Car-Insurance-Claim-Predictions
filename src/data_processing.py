import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

"""
Basic Functions
This module contains functions for data loading, preprocessing
"""
def load_data(train_path, test_path):
    """Load the raw training and test datasets."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def handle_missing_values(df):
    # Identify categorical and numerical columns
    cat_columns = df.select_dtypes(include=['object']).columns
    num_columns = df.select_dtypes(exclude=['object']).columns
    
    missing_cat_count = df[cat_columns].isnull().sum().sum()
    missing_num_count = df[num_columns].isnull().sum().sum()

    print(f"Missing values before imputation: Categorical: {missing_cat_count}, Numerical: {missing_num_count}")

    # Impute categorical columns with mode (most frequent value)
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_columns] = cat_imputer.fit_transform(df[cat_columns])

    # Impute numerical columns with mean
    num_imputer = SimpleImputer(strategy='mean')
    df[num_columns] = num_imputer.fit_transform(df[num_columns])

    # Print missing values after imputation
    missing_cat_count_after = df[cat_columns].isnull().sum().sum()
    missing_num_count_after = df[num_columns].isnull().sum().sum()

    print(f"Missing values after imputation: Categorical: {missing_cat_count_after}, Numerical: {missing_num_count_after}")

    return df

def remove_duplicates(df, verbose=True):
    
    if verbose:
        duplicate_count = df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicate_count}")
    
    df_cleaned = df.drop_duplicates()
    
    if verbose:
        print(f"Dataset shape after removing duplicates: {df_cleaned.shape}")
        duplicate_count_after = df_cleaned.duplicated().sum()
        print(f"Number of duplicate rows after removing rows: {duplicate_count_after}")
    
    return df_cleaned

def Feature_engineering(df):
    """
    Preprocess the insurance dataset:
    - Drop 'policy_id',length, width, height
    - Convert 'Yes'/'No' columns to 1/0
    - Parse max_torque and max_power
    - Create new 'car_volume' feature
    """   
    # 1. Drop unnecessary column
    df.drop(columns=['policy_id'], inplace=True, errors='ignore')

    # 2. Convert boolean columns ('Yes'/'No') to 1/0
    bool_columns = [
        col for col in df.columns 
        if df[col].dtype == 'object' and df[col].nunique() == 2 and 
           sorted(df[col].dropna().unique()) == ['No', 'Yes']
    ]

    for col in bool_columns:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0).astype('int8')

    # 3. Parse max_torque (torque_value and torque_rpm)
    if 'max_torque' in df.columns:
        df[['torque_value', 'torque_rpm']] = df['max_torque'].str.extract(r'(\d+\.?\d*)Nm@(\d+\.?\d*)rpm')
        df['torque_value'] = pd.to_numeric(df['torque_value'], errors='coerce')
        df['torque_rpm'] = pd.to_numeric(df['torque_rpm'], errors='coerce')
        df.drop(columns=['max_torque'], inplace=True)

    if 'max_power' in df.columns:
        df[['power_value', 'power_rpm']] = df['max_power'].str.extract(r'(\d+\.?\d*)bhp@(\d+\.?\d*)rpm')
        df['power_value'] = pd.to_numeric(df['power_value'], errors='coerce')
        df['power_rpm'] = pd.to_numeric(df['power_rpm'], errors='coerce')
        df.drop(columns=['max_power'], inplace=True)

    # Create new 'car_volume' feature
    df['car_volume'] = df['length'] * df['width'] * df['height']
    df.drop(['length', 'width','height'], axis=1, inplace=True)

    return df

def encode_categorical_data(df):
    # 1. Ordinal columns
    ordinal_columns = ['ncap_rating']
    if ordinal_columns:
        ord_enc = OrdinalEncoder()
        for col in ordinal_columns:
            if col in df.columns:
                df[col] = ord_enc.fit_transform(df[[col]])

    # 2. Grouped Ordinal Columns
    group_ordinal_cols = ['age_group', 'car_age_group']
    existing_group_cols = [col for col in group_ordinal_cols if col in df.columns]
    if existing_group_cols:
        group_imputer = SimpleImputer(strategy='most_frequent')
        df[existing_group_cols] = group_imputer.fit_transform(df[existing_group_cols])
        ord_enc = OrdinalEncoder()
        df[existing_group_cols] = ord_enc.fit_transform(df[existing_group_cols])

    # 3. Nominal categorical columns (including make, gear_box, cylinder,airbags)
    categorical_columns = [
        'fuel_type', 'model', 'segment', 'area_cluster',
        'engine_type', 'transmission_type', 'steering_type',
        'rear_brakes_type', 'make', 'gear_box', 'cylinder','airbags'
    ]
    le = LabelEncoder()
    for col in categorical_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))  # Ensure consistent type

    return df


"""
EDA Functions
This module contains functions for exploratory data analysis (EDA)
"""
def plot_categorical_distributions(df, categorical_features, n_cols=2, figsize_per_row=5, suptitle="Categorical Feature Distributions"):
    """
    Plots count plots for selected categorical features in a grid layout.

    Parameters:
        df (pd.DataFrame): DataFrame containing the categorical features.
        categorical_features (list): List of categorical column names to plot.
        n_cols (int): Number of columns in the subplot grid. Defaults to 2.
        figsize_per_row (int): Height of each row in inches. Defaults to 5.
        suptitle (str): Main title of the entire plot. Defaults to 'Categorical Feature Distributions'.
    """
    n_rows = math.ceil(len(categorical_features) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, figsize_per_row * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(categorical_features):
        sns.countplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(f"Count of {col}")
        axes[i].tick_params(axis='x', rotation=45)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle(suptitle, fontsize=18, y=1.02)
    plt.show()

def plot_numerical_vs_target(df, numerical_features, target_col='is_claim'):
    filtered_features = [col for col in numerical_features if col != target_col]
    fig, axes = plt.subplots(nrows=len(filtered_features), ncols=2, 
                             figsize=(15, 4 * len(filtered_features)), 
                             constrained_layout=True)

    palette = {0: '#1f77b4', 1: '#ff7f0e'}

    for i, col in enumerate(filtered_features):
        # Boxplot
        sns.boxplot(ax=axes[i, 0], x=target_col, y=col, data=df)
        axes[i, 0].set_title(f'{col} Distribution by {target_col}')
        axes[i, 0].set_xlabel('Claim Status')
        axes[i, 0].set_ylabel(col)

        # KDE Plot
        sns.kdeplot(ax=axes[i, 1], data=df, x=col, hue=target_col, 
                    palette=palette, common_norm=False, fill=True)
        axes[i, 1].set_title(f'{col} Density by {target_col}')
        axes[i, 1].set_xlabel(col)
        axes[i, 1].set_ylabel('Density')

    plt.suptitle('Numerical Features vs Claim Status Bivariate Analysis', y=1.02, fontsize=16)
    plt.show()

def plot_categorical_vs_target(df, categorical_features, target_col='is_claim'):
   # Create figure with subplots
    fig, axes = plt.subplots(nrows=len(categorical_features), ncols=2, 
                            figsize=(18, 5*len(categorical_features)),
                            constrained_layout=True)

    # Color settings
    palette = {0: '#1f77b4', 1: '#ff7f0e'}  # Blue for no claim, orange for claim
    mean_line_color = '#d62728'  # Red for mean line

    for i, col in enumerate(categorical_features):
        # Claim Rate Bar Plot (Left Column)
        claim_rates = df.groupby(col)[target_col].mean().sort_values()
        claim_rates.plot(kind='bar', ax=axes[i,0], color='#1f77b4')
        axes[i,0].axhline(y=df[target_col].mean(), 
                        color=mean_line_color, linestyle='--', linewidth=2)
        axes[i,0].set_title(f'Claim Rate by {col}', pad=12)
        axes[i,0].set_ylabel('Claim Probability')
        axes[i,0].tick_params(axis='x', rotation=45)
        
        # Stacked Bar Plot (Right Column)
        crosstab = pd.crosstab(df[col], df[target_col])
        crosstab = crosstab.div(crosstab.sum(1), axis=0)  # Normalize
        crosstab.plot(kind='bar', stacked=True, ax=axes[i,1], 
                    color=[palette[0], palette[1]])
        axes[i,1].set_title(f'Normalized Claim Distribution by {col}', pad=12)
        axes[i,1].set_ylabel('Proportion')
        axes[i,1].legend(title='Claim', labels=['No', 'Yes'])
        axes[i,1].tick_params(axis='x', rotation=45)

    plt.suptitle('Categorical Features vs Claim Status Analysis', y=1.02, fontsize=16)
    plt.show()

def plot_boolean_vs_target(df, boolean_features, target_col='is_claim'):
    n_cols = 4
    n_rows = -(-len(boolean_features) // n_cols)  # Ceiling division
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for i, col in enumerate(boolean_features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.barplot(x=col, y=target_col, data=df)
        plt.title(f'Claim Rate by {col}')

    plt.tight_layout()
    plt.suptitle('Boolean Features vs Claim Status', fontsize=16, y=1.02)
    plt.show()

def plot_grouped_age_vs_target(df, policy_age_col='age_of_policyholder', car_age_col='age_of_car', target_col='is_claim'):
    # Create grouped age categories
    df = df.copy()  # Avoid modifying the original DataFrame

    df['age_group'] = pd.cut(df[policy_age_col],
                             bins=[0.25, 0.40, 0.55, 0.70, 0.85, 1.00],
                             labels=['Young', 'Young Adult', 'Adult', 'Middle-Aged', 'Senior'])

    df['car_age_group'] = pd.cut(df[car_age_col],
                                 bins=[0.0, 0.02, 0.06, 0.11, 1.0],
                                 labels=['New (0–0.02)', 'Slightly Used', 'Moderate', 'Old'])

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Policyholder Age Group vs Claim Rate
    sns.barplot(data=df, x='age_group', y=target_col, ax=axes[0])
    axes[0].set_title('Claim Rate by Policyholder Age')
    axes[0].set_xlabel('Policyholder Age Group')
    axes[0].set_ylabel('Claim Rate')

    # Car Age Group vs Claim Rate
    sns.barplot(data=df, x='car_age_group', y=target_col, ax=axes[1])
    axes[1].set_title('Claim Rate by Car Age')
    axes[1].set_xlabel('Car Age Group')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.suptitle("Claim Rates by Age Groupings", fontsize=16, y=1.03)
    plt.show()

def plot_grouped_means_heatmap(df, num_features, target_col='is_claim'):
    """
    Plots a heatmap of mean values of numerical features grouped by the target column.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        num_features (list): List of numerical feature column names.
        target_col (str): Name of the target column to group by (default is 'is_claim').
    """
    grouped_means = df.groupby(target_col)[num_features].mean().T

    plt.figure(figsize=(15, 10))
    sns.heatmap(grouped_means, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(f'Mean of Numerical Features by {target_col.capitalize()}')
    plt.ylabel('Numerical Features')
    plt.xlabel(target_col.capitalize())
    plt.tight_layout()
    plt.show()

def deep_dive_categorical_analysis(df):
    """
    Generates a multi-plot analysis of categorical features including:
    1. Claim rate by area cluster (heatmap)
    2. Segment × engine type interaction (heatmap)
    3. Power distribution for top 15 car models by claim status (boxplot)
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing columns:
            'area_cluster', 'segment', 'engine_type', 'model', 'power_value', and 'is_claim'
    """
    plt.figure(figsize=(16, 12))
    gs = plt.GridSpec(2, 2, height_ratios=[1.5, 1])

    # 1. Area Cluster Heatmap (Top Left)
    plt.subplot(gs[0, 0])
    area_claim = df.groupby('area_cluster')['is_claim'].mean().sort_values()
    sns.heatmap(area_claim.to_frame(), annot=True, fmt='.2%', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Claim Rate'})
    plt.title('Claim Rate by Area Cluster', pad=12)
    plt.yticks(rotation=0)
    plt.xlabel('')

    # 2. Segment vs Engine Type (Top Right)
    plt.subplot(gs[0, 1])
    segment_engine = df.groupby(['segment', 'engine_type'])['is_claim'].mean().unstack()
    sns.heatmap(segment_engine, annot=True, fmt='.1%', cmap='YlOrBr', 
                cbar_kws={'label': 'Claim Rate'})
    plt.title('Segment × Engine Type Interaction', pad=12)
    plt.ylabel('')

    # 3. Model Risk Ranking (Bottom)
    plt.subplot(gs[1, :])
    top_models = df['model'].value_counts().nlargest(15).index
    model_data = df[df['model'].isin(top_models)]

    # Ensure the palette keys match the actual dtype of 'is_claim'
    unique_classes = model_data['is_claim'].unique()
    palette = {cls: color for cls, color in zip(sorted(unique_classes), ['#3498db', '#e74c3c'])}
    
    sns.boxplot(data=model_data, x='model', y='power_value', hue='is_claim',
                palette=palette, order=top_models)
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 15 Models: Power Distribution by Claim Status', pad=12)
    plt.xlabel('Model')
    plt.ylabel('Power (bhp)')
    plt.legend(title='Claim', labels=['No', 'Yes'])

    plt.tight_layout()
    plt.suptitle('Deep Dive: Categorical Features Analysis', y=1.02, fontsize=16)
    plt.show()

def plot_boolean_feature_claim_rates(df, bool_features, target_col='is_claim'):
    """
    Plots claim rate comparison for boolean features (0/1) in relation to the target variable.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        bool_features (list): List of boolean feature column names (binary 0/1).
        target_col (str): Name of the target column (default is 'is_claim').
    """
    results = []
    
    for feature in bool_features:
        temp = df.groupby(feature)[target_col].mean().reset_index()
        temp.columns = ['Value', 'Claim Rate']
        temp['Feature'] = feature
        results.append(temp)

    result_df = pd.concat(results, ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=result_df, x='Feature', y='Claim Rate', hue='Value', palette='Set2')
    
    plt.title('Claim Rate by Safety Feature (0 = No, 1 = Yes)')
    plt.ylabel('Claim Rate')
    plt.xlabel('Feature')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.legend(title='Feature Present')
    plt.tight_layout()
    plt.show()

""" 
Deep Preprocessing Functions
This module contains functions for deep preprocessing of the dataset. Outliers Handling,
Feature Scaling, and Class Imbalance Handling.
"""
def handle_outliers(df, dataset_name='Dataset', show_plots=True):
    """ outlier handling with visual """
    
    # Step 1: Select only original numeric columns (excluding future log transforms)
    original_numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Step 2: Plot BEFORE state if requested
    if show_plots:
        plot_boxplots(df[original_numeric_cols], 
                     title=f"{dataset_name} - BEFORE Outlier Removal")
    return df

def plot_boxplots(data, title):
    """Helper function to plot boxplots in grid"""
    n_cols = 4
    n_rows = int(np.ceil(len(data.columns)/n_cols))
    
    plt.figure(figsize=(20, 5*n_rows))
    for i, col in enumerate(data.columns):
        plt.subplot(n_rows, n_cols, i+1)
        sns.boxplot(x=data[col])
        plt.title(col)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.show()

def scale_important_features(df, scaler=None):
    """
    Scale important numeric features using StandardScaler.
    """
    important_numeric = [
        'policy_tenure', 'age_of_policyholder', 'population_density', 'displacement', 'gross_weight',
        'ncap_rating', 'turning_radius', 'age_of_car', 'airbags',
        'torque_value', 'torque_rpm', 'power_value', 'power_rpm'
    ]
    
    # Filter only existing columns (if a column was dropped earlier)
    existing_cols = [col for col in important_numeric if col in df.columns]
    if scaler is None:
        scaler = StandardScaler()
        df[existing_cols] = scaler.fit_transform(df[existing_cols])
        return df, scaler
    else:
        df[existing_cols] = scaler.transform(df[existing_cols])
        return df
    
