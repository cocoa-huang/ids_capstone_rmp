"""
Questions 7-10: Modeling
Authors: Eric Huang, Bruce Zhang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
from sklearn.linear_model import LinearRegression, LogisticRegression
warnings.filterwarnings('ignore')


# set seed number
SEED = 11269807 
np.random.seed(SEED)

# Load processed data
print("Loading data...")
df = pd.read_csv("../data/rmpCapstoneProcessed.csv")


run_q7 = False
run_q8 = False
run_q9 = False
run_q10 = False

# ============================================================================
# QUESTION 7: Regression - Rating from Numerical Features
# ============================================================================
# defined features from data

if run_q7: 
    num_features = [
        'difficulty', 'rating_count', 'pepper', 
        'will_retake_pct', 'online_count', 'male', 'female'
    ]

    # Select features and target
    X = df[num_features].copy()
    y = df['rating'].copy()

    # collinearity analysis
    print("\n" + "=" * 80)
    print("COLLINEARITY ANALYSIS")
    print("=" * 80)

    # Correlation with target (rating)
    print("\n--- Correlation with Target (rating) ---")
    corr_with_target = X.corrwith(y).sort_values(ascending=False)
    print(corr_with_target.round(4))

    # Correlation matrix among predictors
    print("\n--- Correlation Matrix Among Predictors ---")
    corr_matrix = X.corr()
    print(corr_matrix.round(3))

    # Variance Inflation Factor (VIF)
    print("\n--- Variance Inflation Factor (VIF) ---")

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)
    print(vif_data.to_string(index=False))

    # Did not drop any features
    # VIF for will_retake_pct is silghtly over 10, and that it is highly correlated with rating
    # high VIF might be because of data imputation

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # Fit Linear Regression Model
    print("\n" + "=" * 80)
    print("MODEL TRAINING")
    print("=" * 80)

    # Fit linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)


    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Display results
    print("\n--- Model Performance ---")
    print(f"Training Set:")
    print(f"  R^2 = {train_r2:.4f}")
    print(f"  RMSE = {train_rmse:.4f}")
    print(f"\nTest Set:")
    print(f"  R^2 = {test_r2:.4f}")
    print(f"  RMSE = {test_rmse:.4f}")

    # Check for overfitting
    r2_diff = abs(train_r2 - test_r2)
    if r2_diff < 0.05:
        print(f"\n✓ Good generalization (R² difference: {r2_diff:.4f})")
    else:
        print(f"\n⚠️  Potential overfitting (R² difference: {r2_diff:.4f})")

    # Feature Importance Analysis
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    # Raw coefficients (for reference)
    print("\n--- Raw Coefficients ---")
    coef_df = pd.DataFrame({
        'Feature': num_features,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    print(coef_df.to_string(index=False))

    # Standardized coefficients (Beta weights) - for fair comparison
    print("\n--- Standardized Coefficients (Beta Weights) ---")
    print("(These allow fair comparison across different scales)")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit model on standardized data
    model_scaled = LinearRegression()
    model_scaled.fit(X_train_scaled, y_train)

    # Get standardized coefficients
    std_coef_df = pd.DataFrame({
        'Feature': num_features,
        'Std_Coefficient': model_scaled.coef_,
        'Abs_Std_Coefficient': np.abs(model_scaled.coef_)
    }).sort_values('Abs_Std_Coefficient', ascending=False)
    print(std_coef_df.to_string(index=False))

    # Identify most predictive feature
    most_predictive = std_coef_df.iloc[0]['Feature']
    most_predictive_beta = std_coef_df.iloc[0]['Std_Coefficient']

    print(f"MOST STRONGLY PREDICTIVE FEATURE: {most_predictive}")
    print(f"   Standardized coefficient (β): {most_predictive_beta:.4f}")

    # visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Predicted vs Actual (Test Set)
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.3, s=10, color='steelblue')
    axes[0, 0].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Rating', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Rating', fontsize=11)
    axes[0, 0].set_title(f'Predicted vs Actual Rating (Test Set)\nR² = {test_r2:.4f}, RMSE = {test_rmse:.4f}', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Residuals Plot
    residuals = y_test - y_test_pred
    axes[0, 1].scatter(y_test_pred, residuals, alpha=0.3, s=10, color='steelblue')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Rating', fontsize=11)
    axes[0, 1].set_ylabel('Residuals', fontsize=11)
    axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Raw Coefficients
    axes[1, 0].barh(coef_df['Feature'], coef_df['Coefficient'], color='steelblue')
    axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[1, 0].set_xlabel('Coefficient Value', fontsize=11)
    axes[1, 0].set_title('Raw Coefficients', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # Panel 4: Standardized Coefficients (highlight most predictive)
    colors = ['red' if feat == most_predictive else 'steelblue' 
              for feat in std_coef_df['Feature']]
    axes[1, 1].barh(std_coef_df['Feature'], std_coef_df['Std_Coefficient'], color=colors)
    axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[1, 1].set_xlabel('Standardized Coefficient (β)', fontsize=11)
    axes[1, 1].set_title(f'Standardized Coefficients\nMost Predictive: {most_predictive} (β = {most_predictive_beta:.4f})', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('../figures/Q7_regression_numerical.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# QUESTION 8: Regression - Rating from Tags
# ============================================================================
if run_q8: 
    # extract tags cols
    tag_features = [col for col in df.columns if col.endswith('_norm')]
    tag_features.sort() 

    X_tags = df[tag_features].copy()
    y_tags = df['rating'].copy()
    

    print("\n" + "=" * 80)
    print("COLLINEARITY ANALYSIS - TAGS")
    print("=" * 80)

    # 2a. Correlation with target (rating)
    print("\n--- Correlation with Target (rating) ---")
    print("Top 10 most correlated tags:")
    corr_with_rating = X_tags.corrwith(y_tags).sort_values(ascending=False)
    print(corr_with_rating.head(10).round(4))

    print("\nBottom 5 (least/negatively correlated):")
    print(corr_with_rating.tail(5).round(4))

    # 2b. Check correlation matrix among tags
    print("\n--- Checking for High Correlations Among Tags ---")
    corr_matrix_tags = X_tags.corr()

    # Find high correlations (|r| > 0.7)
    high_corr_tags = []
    for i in range(len(corr_matrix_tags.columns)):
        for j in range(i+1, len(corr_matrix_tags.columns)):
            corr_val = corr_matrix_tags.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_tags.append({
                    'Tag_1': corr_matrix_tags.columns[i],
                    'Tag_2': corr_matrix_tags.columns[j],
                    'Correlation': corr_val
                })

    if high_corr_tags:
        print(f"Found {len(high_corr_tags)} tag pairs with |r| > 0.7:")
        for pair in high_corr_tags:
            print(f"  {pair['Tag_1']:<35} <-> {pair['Tag_2']:<35}: {pair['Correlation']:.3f}")
    else:
        print("✓ No tag pairs with |r| > 0.7")

    # 2c. VIF Analysis for tags
    print("\n--- Variance Inflation Factor (VIF) for Tags ---")
    print("(This may take a moment with 20 features...)")

    vif_tags = pd.DataFrame()
    vif_tags["Tag"] = X_tags.columns
    vif_tags["VIF"] = [variance_inflation_factor(X_tags.values, i) for i in range(X_tags.shape[1])]
    vif_tags = vif_tags.sort_values('VIF', ascending=False).reset_index(drop=True)

    print("\nTop 10 highest VIF:")
    print(vif_tags.head(10).to_string(index=False))

    # Check for problematic VIF
    high_vif_tags = vif_tags[vif_tags['VIF'] > 10]
    if len(high_vif_tags) > 0:
        print(f"\n⚠️  WARNING: {len(high_vif_tags)} tag(s) with VIF > 10")
        print(high_vif_tags.to_string(index=False))
    else:
        print("\n✓ All tag VIF < 10 (acceptable multicollinearity)")

    # train test split\
    print("\n" + "=" * 80)
    print("TRAIN/TEST SPLIT - TAGS")
    print("=" * 80)

    # Use same random seed for consistency
    X_tags_train, X_tags_test, y_tags_train, y_tags_test = train_test_split(
        X_tags, y_tags, test_size=0.2, random_state=SEED
    )

    print("\n" + "=" * 80)
    print("MODEL TRAINING - TAGS")
    print("=" * 80)

    # Fit linear regression
    model_tags = LinearRegression()
    model_tags.fit(X_tags_train, y_tags_train)

    print("✓ Model fitted successfully")

    # Make predictions
    y_tags_train_pred = model_tags.predict(X_tags_train)
    y_tags_test_pred = model_tags.predict(X_tags_test)

    # Calculate metrics
    train_r2_tags = r2_score(y_tags_train, y_tags_train_pred)
    train_rmse_tags = np.sqrt(mean_squared_error(y_tags_train, y_tags_train_pred))
    test_r2_tags = r2_score(y_tags_test, y_tags_test_pred)
    test_rmse_tags = np.sqrt(mean_squared_error(y_tags_test, y_tags_test_pred))

    # Display results
    print("\n--- Model Performance (Tags) ---")
    print(f"Training Set:")
    print(f"  R² = {train_r2_tags:.4f}")
    print(f"  RMSE = {train_rmse_tags:.4f}")
    print(f"\nTest Set:")
    print(f"  R² = {test_r2_tags:.4f}")
    print(f"  RMSE = {test_rmse_tags:.4f}")

    # Check for overfitting
    r2_diff_tags = abs(train_r2_tags - test_r2_tags)
    if r2_diff_tags < 0.05:
        print(f"\n✓ Good generalization (R² difference: {r2_diff_tags:.4f})")
    else:
        print(f"\n⚠️  Potential overfitting (R² difference: {r2_diff_tags:.4f})")
        
        
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS - TAGS")
    print("=" * 80)

    # 5a. Raw coefficients (for reference)
    print("\n--- Raw Coefficients (Top 10) ---")
    coef_tags_df = pd.DataFrame({
        'Tag': tag_features,
        'Coefficient': model_tags.coef_,
        'Abs_Coefficient': np.abs(model_tags.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)

    print(coef_tags_df.head(10).to_string(index=False))

    # 5b. Standardized coefficients (Beta weights) - for fair comparison
    print("\n--- Standardized Coefficients (Beta Weights) ---")
    print("(Fair comparison across different tag scales)")

    # Standardize tag features
    scaler_tags = StandardScaler()
    X_tags_train_scaled = scaler_tags.fit_transform(X_tags_train)
    X_tags_test_scaled = scaler_tags.transform(X_tags_test)

    # Fit model on standardized data
    model_tags_scaled = LinearRegression()
    model_tags_scaled.fit(X_tags_train_scaled, y_tags_train)

    # Get standardized coefficients
    std_coef_tags_df = pd.DataFrame({
        'Tag': tag_features,
        'Std_Coefficient': model_tags_scaled.coef_,
        'Abs_Std_Coefficient': np.abs(model_tags_scaled.coef_)
    }).sort_values('Abs_Std_Coefficient', ascending=False)

    print("\nTop 10 Most Predictive Tags:")
    print(std_coef_tags_df.head(10).to_string(index=False))

    print("\nBottom 5 Least Predictive Tags:")
    print(std_coef_tags_df.tail(5).to_string(index=False))

    # Identify most predictive tag
    most_predictive_tag = std_coef_tags_df.iloc[0]['Tag']
    most_predictive_tag_beta = std_coef_tags_df.iloc[0]['Std_Coefficient']

    print(f"\n🎯 MOST STRONGLY PREDICTIVE TAG: {most_predictive_tag}")
    print(f"   Standardized coefficient (β): {most_predictive_tag_beta:.4f}")
    
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION - TAGS")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Predicted vs Actual (Test Set)
    axes[0, 0].scatter(y_tags_test, y_tags_test_pred, alpha=0.3, s=10, color='steelblue')
    axes[0, 0].plot([y_tags_test.min(), y_tags_test.max()], 
                    [y_tags_test.min(), y_tags_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Rating', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Rating', fontsize=11)
    axes[0, 0].set_title(f'Predicted vs Actual Rating (Test Set)\nR² = {test_r2_tags:.4f}, RMSE = {test_rmse_tags:.4f}', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Residuals Plot
    residuals_tags = y_tags_test - y_tags_test_pred
    axes[0, 1].scatter(y_tags_test_pred, residuals_tags, alpha=0.3, s=10, color='steelblue')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Rating', fontsize=11)
    axes[0, 1].set_ylabel('Residuals', fontsize=11)
    axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Top 10 Tags (Raw Coefficients)
    top10_tags = coef_tags_df.head(10)
    colors_raw = ['red' if tag == most_predictive_tag else 'steelblue' 
                for tag in top10_tags['Tag']]
    axes[1, 0].barh(range(len(top10_tags)), top10_tags['Coefficient'], color=colors_raw)
    axes[1, 0].set_yticks(range(len(top10_tags)))
    axes[1, 0].set_yticklabels([tag.replace('_norm', '') for tag in top10_tags['Tag']], fontsize=9)
    axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[1, 0].set_xlabel('Coefficient Value', fontsize=11)
    axes[1, 0].set_title('Top 10 Tags - Raw Coefficients', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].invert_yaxis()

    # Panel 4: Top 10 Tags (Standardized Coefficients)
    top10_std = std_coef_tags_df.head(10)
    colors_std = ['red' if tag == most_predictive_tag else 'steelblue' 
                for tag in top10_std['Tag']]
    axes[1, 1].barh(range(len(top10_std)), top10_std['Std_Coefficient'], color=colors_std)
    axes[1, 1].set_yticks(range(len(top10_std)))
    axes[1, 1].set_yticklabels([tag.replace('_norm', '') for tag in top10_std['Tag']], fontsize=9)
    axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[1, 1].set_xlabel('Standardized Coefficient (β)', fontsize=11)
    axes[1, 1].set_title(f'Top 10 Tags - Standardized Coefficients\nMost Predictive: {most_predictive_tag.replace("_norm", "")} (β = {most_predictive_tag_beta:.4f})', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('../figures/Q8_regression_tags.png', dpi=300, bbox_inches='tight')
    print("✓ Figure saved: ../figures/Q8_regression_tags.png")
    plt.show()
    
# ============================================================================
# QUESTION 9: Regression - Predicting Difficulty from Tags
# ============================================================================

# ============================================================================
# Step 1: Prepare Data
# ============================================================================
if run_q9:
    tag_features = [col for col in df.columns if col.endswith('_norm')]
    # Use same tag features, but predict difficulty instead of rating
    X_diff = df[tag_features].copy()
    y_diff = df['difficulty'].copy()

    print(f"X_diff shape: {X_diff.shape}")
    print(f"y_diff shape: {y_diff.shape}")

    # Check for missing values
    missing_diff = y_diff.isnull().sum()
    print(f"Missing values in difficulty: {missing_diff}")

    # ============================================================================
    # Step 2: Collinearity (Quick Check)
    # ============================================================================
    print("\n--- Collinearity Analysis ---")
    print("Note: Tag collinearity already analyzed in Q8 (all VIF < 2)")

    # Correlation with difficulty (new target)
    print("\nTop 10 tags correlated with difficulty:")
    corr_with_diff = X_diff.corrwith(y_diff).sort_values(ascending=False)
    print(corr_with_diff.head(10).round(4))

    print("\nBottom 5 (negatively correlated):")
    print(corr_with_diff.tail(5).round(4))

    # ============================================================================
    # Step 3: Train/Test Split
    # ============================================================================
    print("\n" + "=" * 80)
    print("TRAIN/TEST SPLIT")
    print("=" * 80)

    X_diff_train, X_diff_test, y_diff_train, y_diff_test = train_test_split(
        X_diff, y_diff, test_size=0.2, random_state=SEED
    )

    print(f"Train set: {len(X_diff_train):,} professors ({len(X_diff_train)/len(X_diff)*100:.1f}%)")
    print(f"Test set:  {len(X_diff_test):,} professors ({len(X_diff_test)/len(X_diff)*100:.1f}%)")

    # ============================================================================
    # Step 4: Fit Model
    # ============================================================================
    print("\n" + "=" * 80)
    print("MODEL TRAINING")
    print("=" * 80)

    # Fit linear regression
    model_diff = LinearRegression()
    model_diff.fit(X_diff_train, y_diff_train)

    print("✓ Model fitted successfully")

    # Make predictions
    y_diff_train_pred = model_diff.predict(X_diff_train)
    y_diff_test_pred = model_diff.predict(X_diff_test)

    # Calculate metrics
    train_r2_diff = r2_score(y_diff_train, y_diff_train_pred)
    train_rmse_diff = np.sqrt(mean_squared_error(y_diff_train, y_diff_train_pred))
    test_r2_diff = r2_score(y_diff_test, y_diff_test_pred)
    test_rmse_diff = np.sqrt(mean_squared_error(y_diff_test, y_diff_test_pred))

    # Display results
    print("\n--- Model Performance ---")
    print(f"Training Set:")
    print(f"  R² = {train_r2_diff:.4f}")
    print(f"  RMSE = {train_rmse_diff:.4f}")
    print(f"\nTest Set:")
    print(f"  R² = {test_r2_diff:.4f}")
    print(f"  RMSE = {test_rmse_diff:.4f}")

    # Check for overfitting
    r2_diff_overfitting = abs(train_r2_diff - test_r2_diff)
    if r2_diff_overfitting < 0.05:
        print(f"\n✓ Good generalization (R² difference: {r2_diff_overfitting:.4f})")
    else:
        print(f"\n⚠️  Potential overfitting (R² difference: {r2_diff_overfitting:.4f})")

    # ============================================================================
    # Step 5: Feature Importance (Standardized Coefficients)
    # ============================================================================
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    # Raw coefficients
    coef_diff_df = pd.DataFrame({
        'Tag': tag_features,
        'Coefficient': model_diff.coef_,
        'Abs_Coefficient': np.abs(model_diff.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)

    print("\n--- Raw Coefficients (Top 10) ---")
    print(coef_diff_df.head(10).to_string(index=False))

    # Standardized coefficients
    scaler_diff = StandardScaler()
    X_diff_train_scaled = scaler_diff.fit_transform(X_diff_train)
    X_diff_test_scaled = scaler_diff.transform(X_diff_test)

    model_diff_scaled = LinearRegression()
    model_diff_scaled.fit(X_diff_train_scaled, y_diff_train)

    std_coef_diff_df = pd.DataFrame({
        'Tag': tag_features,
        'Std_Coefficient': model_diff_scaled.coef_,
        'Abs_Std_Coefficient': np.abs(model_diff_scaled.coef_)
    }).sort_values('Abs_Std_Coefficient', ascending=False)

    print("\n--- Standardized Coefficients (Top 10) ---")
    print(std_coef_diff_df.head(10).to_string(index=False))

    # Most predictive tag
    most_predictive_diff_tag = std_coef_diff_df.iloc[0]['Tag']
    most_predictive_diff_beta = std_coef_diff_df.iloc[0]['Std_Coefficient']

    print(f"\n🎯 MOST STRONGLY PREDICTIVE TAG: {most_predictive_diff_tag}")
    print(f"   Standardized coefficient (β): {most_predictive_diff_beta:.4f}")

    # ============================================================================
    # Step 6: Create Visualization
    # ============================================================================
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Predicted vs Actual
    axes[0, 0].scatter(y_diff_test, y_diff_test_pred, alpha=0.3, s=10, color='steelblue')
    axes[0, 0].plot([y_diff_test.min(), y_diff_test.max()], 
                    [y_diff_test.min(), y_diff_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Difficulty', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Difficulty', fontsize=11)
    axes[0, 0].set_title(f'Predicted vs Actual Difficulty (Test Set)\nR² = {test_r2_diff:.4f}, RMSE = {test_rmse_diff:.4f}', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Residuals
    residuals_diff = y_diff_test - y_diff_test_pred
    axes[0, 1].scatter(y_diff_test_pred, residuals_diff, alpha=0.3, s=10, color='steelblue')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Difficulty', fontsize=11)
    axes[0, 1].set_ylabel('Residuals', fontsize=11)
    axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Top 10 Raw Coefficients
    top10_diff = coef_diff_df.head(10)
    colors_raw_diff = ['red' if tag == most_predictive_diff_tag else 'steelblue' 
                    for tag in top10_diff['Tag']]
    axes[1, 0].barh(range(len(top10_diff)), top10_diff['Coefficient'], color=colors_raw_diff)
    axes[1, 0].set_yticks(range(len(top10_diff)))
    axes[1, 0].set_yticklabels([tag.replace('_norm', '') for tag in top10_diff['Tag']], fontsize=9)
    axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[1, 0].set_xlabel('Coefficient Value', fontsize=11)
    axes[1, 0].set_title('Top 10 Tags - Raw Coefficients', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].invert_yaxis()

    # Panel 4: Top 10 Standardized Coefficients
    top10_std_diff = std_coef_diff_df.head(10)
    colors_std_diff = ['red' if tag == most_predictive_diff_tag else 'steelblue' 
                    for tag in top10_std_diff['Tag']]
    axes[1, 1].barh(range(len(top10_std_diff)), top10_std_diff['Std_Coefficient'], color=colors_std_diff)
    axes[1, 1].set_yticks(range(len(top10_std_diff)))
    axes[1, 1].set_yticklabels([tag.replace('_norm', '') for tag in top10_std_diff['Tag']], fontsize=9)
    axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[1, 1].set_xlabel('Standardized Coefficient (β)', fontsize=11)
    axes[1, 1].set_title(f'Top 10 Tags - Standardized Coefficients\nMost Predictive: {most_predictive_diff_tag.replace("_norm", "")} (β = {most_predictive_diff_beta:.4f})', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('../figures/Q9_regression_difficulty_tags.png', dpi=300, bbox_inches='tight')
    print("✓ Figure saved: ../figures/Q9_regression_difficulty_tags.png")
    plt.show()
    
# ============================================================================
# QUESTION 10: Classification - Predicting Pepper
# ============================================================================
if run_q10:
    
    num_features = [
        'difficulty', 'rating_count', 
        'will_retake_pct', 'online_count', 'male', 'female'
    ]
    
    tag_features = [col for col in df.columns if col.endswith('_norm')]
    all_features = num_features + tag_features
    
    X_pepper = df[all_features].copy()
    y_pepper = df['pepper'].copy()
    
    class_counts = y_pepper.value_counts().sort_index()
    class_pcts = y_pepper.value_counts(normalize=True).sort_index() * 100

    print("\nPepper Distribution:")
    print(f"  No Pepper (0): {class_counts[0]:,} ({class_pcts[0]:.2f}%)")
    print(f"  Pepper (1):    {class_counts[1]:,} ({class_pcts[1]:.2f}%)")

    # Calculate imbalance ratio
    imbalance_ratio = class_counts[0] / class_counts[1]
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1 (no pepper : pepper)")

   
    print("\n" + "=" * 80)
    print("TRAIN/TEST SPLIT (STRATIFIED)")
    print("=" * 80)

    # Use stratified split to maintain class balance in both sets
    X_pepper_train, X_pepper_test, y_pepper_train, y_pepper_test = train_test_split(
        X_pepper, y_pepper, 
        test_size=0.2, 
        random_state=SEED,
        stratify=y_pepper  # KEY: Maintains class distribution
    )

    print(f"Train set: {len(X_pepper_train):,} professors ({len(X_pepper_train)/len(X_pepper)*100:.1f}%)")
    print(f"Test set:  {len(X_pepper_test):,} professors ({len(X_pepper_test)/len(X_pepper)*100:.1f}%)")

    # Verify stratification worked
    print("\nClass Distribution in Train Set:")
    train_dist = y_pepper_train.value_counts(normalize=True).sort_index() * 100
    print(f"  No Pepper (0): {train_dist[0]:.2f}%")
    print(f"  Pepper (1):    {train_dist[1]:.2f}%")

    print("\nClass Distribution in Test Set:")
    test_dist = y_pepper_test.value_counts(normalize=True).sort_index() * 100
    print(f"  No Pepper (0): {test_dist[0]:.2f}%")
    print(f"  Pepper (1):    {test_dist[1]:.2f}%")

    print("\n✓ Stratified split maintains class balance in both sets")
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING - LOGISTIC REGRESSION")
    print("=" * 80)

    # Import additional metrics for classification
    from sklearn.metrics import (
        roc_auc_score, roc_curve, confusion_matrix, 
        classification_report, accuracy_score
    )

    # Fit Logistic Regression with balanced class weights
    # class_weight='balanced' automatically adjusts weights inversely proportional to class frequencies
    model_pepper = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
        class_weight='balanced'  # KEY: Addresses class imbalance
    )

    model_pepper.fit(X_pepper_train, y_pepper_train)
    print("✓ Logistic Regression fitted successfully with balanced class weights")

    # Make predictions
    y_pepper_train_pred = model_pepper.predict(X_pepper_train)
    y_pepper_test_pred = model_pepper.predict(X_pepper_test)

    # Get probability predictions for AUC calculation
    y_pepper_train_proba = model_pepper.predict_proba(X_pepper_train)[:, 1]
    y_pepper_test_proba = model_pepper.predict_proba(X_pepper_test)[:, 1]

    # ============================================================================
    # Step 5: Model Evaluation
    # ============================================================================
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE")
    print("=" * 80)

    # Accuracy
    train_accuracy = accuracy_score(y_pepper_train, y_pepper_train_pred)
    test_accuracy = accuracy_score(y_pepper_test, y_pepper_test_pred)

    print(f"\nAccuracy:")
    print(f"  Training: {train_accuracy:.4f}")
    print(f"  Test:     {test_accuracy:.4f}")

    # AUC-ROC (Required by spec)
    train_auc = roc_auc_score(y_pepper_train, y_pepper_train_proba)
    test_auc = roc_auc_score(y_pepper_test, y_pepper_test_proba)

    print(f"\nAUC-ROC:")
    print(f"  Training: {train_auc:.4f}")
    print(f"  Test:     {test_auc:.4f}")

    # Confusion Matrix (Test Set)
    cm = confusion_matrix(y_pepper_test, y_pepper_test_pred)
    print(f"\nConfusion Matrix (Test Set):")
    print(f"                Predicted")
    print(f"                No    Yes")
    print(f"Actual  No   {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"        Yes  {cm[1,0]:6d} {cm[1,1]:6d}")

    # Classification Report (Test Set)
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_pepper_test, y_pepper_test_pred, 
                            target_names=['No Pepper', 'Pepper']))

    # Calculate ROC curve for visualization
    fpr, tpr, thresholds = roc_curve(y_pepper_test, y_pepper_test_proba)

    print(f"\n🎯 PRIMARY METRIC (Required): AUC-ROC = {test_auc:.4f}")
    
    
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    # Get coefficients from logistic regression
    coef_pepper_df = pd.DataFrame({
        'Feature': all_features,
        'Coefficient': model_pepper.coef_[0],
        'Abs_Coefficient': np.abs(model_pepper.coef_[0])
    }).sort_values('Abs_Coefficient', ascending=False)

    print("\n--- Top 15 Most Predictive Features (Raw Coefficients) ---")
    print(coef_pepper_df.head(15).to_string(index=False))

    # Standardized coefficients for fair comparison
    scaler_pepper = StandardScaler()
    X_pepper_train_scaled = scaler_pepper.fit_transform(X_pepper_train)
    X_pepper_test_scaled = scaler_pepper.transform(X_pepper_test)

    model_pepper_scaled = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
        class_weight='balanced'
    )
    model_pepper_scaled.fit(X_pepper_train_scaled, y_pepper_train)

    std_coef_pepper_df = pd.DataFrame({
        'Feature': all_features,
        'Std_Coefficient': model_pepper_scaled.coef_[0],
        'Abs_Std_Coefficient': np.abs(model_pepper_scaled.coef_[0])
    }).sort_values('Abs_Std_Coefficient', ascending=False)

    print("\n--- Top 15 Most Predictive Features (Standardized) ---")
    print(std_coef_pepper_df.head(15).to_string(index=False))

    most_predictive_pepper = std_coef_pepper_df.iloc[0]['Feature']
    most_predictive_pepper_beta = std_coef_pepper_df.iloc[0]['Std_Coefficient']

    print(f"\n🎯 MOST STRONGLY PREDICTIVE FEATURE: {most_predictive_pepper}")
    print(f"   Standardized coefficient (β): {most_predictive_pepper_beta:.4f}")
    
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: ROC Curve
    axes[0, 0].plot(fpr, tpr, color='steelblue', lw=2, 
                    label=f'ROC Curve (AUC = {test_auc:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', 
                    label='Random Classifier (AUC = 0.50)')
    axes[0, 0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0, 0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0, 0].set_title('ROC Curve - Pepper Classification', fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Confusion Matrix Heatmap
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0, 1],
                xticklabels=['No Pepper', 'Pepper'],
                yticklabels=['No Pepper', 'Pepper'])
    axes[0, 1].set_xlabel('Predicted', fontsize=11)
    axes[0, 1].set_ylabel('Actual', fontsize=11)
    axes[0, 1].set_title(f'Confusion Matrix\nAccuracy = {test_accuracy:.4f}', 
                        fontsize=12, fontweight='bold')

    # Panel 3: Top 15 Raw Coefficients
    top15_raw = coef_pepper_df.head(15)
    colors_raw_pepper = ['red' if feat == most_predictive_pepper else 'steelblue' 
                        for feat in top15_raw['Feature']]
    axes[1, 0].barh(range(len(top15_raw)), top15_raw['Coefficient'], color=colors_raw_pepper)
    axes[1, 0].set_yticks(range(len(top15_raw)))
    axes[1, 0].set_yticklabels([f.replace('_norm', '') for f in top15_raw['Feature']], fontsize=8)
    axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[1, 0].set_xlabel('Coefficient Value', fontsize=11)
    axes[1, 0].set_title('Top 15 Features - Raw Coefficients', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].invert_yaxis()

    # Panel 4: Top 15 Standardized Coefficients
    top15_std_pepper = std_coef_pepper_df.head(15)
    colors_std_pepper = ['red' if feat == most_predictive_pepper else 'steelblue' 
                        for feat in top15_std_pepper['Feature']]
    axes[1, 1].barh(range(len(top15_std_pepper)), top15_std_pepper['Std_Coefficient'], 
                    color=colors_std_pepper)
    axes[1, 1].set_yticks(range(len(top15_std_pepper)))
    axes[1, 1].set_yticklabels([f.replace('_norm', '') for f in top15_std_pepper['Feature']], fontsize=8)
    axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[1, 1].set_xlabel('Standardized Coefficient (β)', fontsize=11)
    axes[1, 1].set_title(f'Top 15 Features - Standardized\nMost Predictive: {most_predictive_pepper.replace("_norm", "")} (β = {most_predictive_pepper_beta:.4f})', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('../figures/Q10_classification_pepper.png', dpi=300, bbox_inches='tight')
    print("✓ Figure saved: ../figures/Q10_classification_pepper.png")
    plt.show()
    
print("Hello")