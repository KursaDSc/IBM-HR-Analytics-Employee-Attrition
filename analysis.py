import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.ensemble import RandomForestRegressor

# ### Realize ANOVA analysis on categorical variables :
# #-----------------------------------------------------------
# # Create a formula to perform ANOVA analysis
# def anova_analysis(cat_cols, target_col):
#     anova_table = []
#     for col in cat_cols:
#         formula = 'target_col ~ df.loc[:,col]'

#         # Building the ANOVA model
#         model = ols(formula, data=df).fit()

#         # Applying ANOVA analysis
#         anova_table.append(sm.stats.anova_lm(model, type=2))
#         # Printing ANOVA results
#     # an_tbl = pd.DataFrame(anova_table)
#     print(anova_table)
# #-----------------------------------------------------------


### Perform feature importance ranking analysis of variables using a Random Forest algorithm
#-----------------------------------------------------------
# Separate the features (X) and target variable (y)
def fi_analysis(df, target):
    X = df.drop(target, axis=1)
    y = df[target]

    # Create a Random Forest model
    rf = RandomForestRegressor()
    rf.fit(X, y)

    # Get the feature importances
    importances = rf.feature_importances_

    # Create a dataframe with feature names and importances
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

    # Sort the dataframe by importance values in descending order
    feature_importances = feature_importances.sort_values('Importance', ascending=False)
    # Print the feature importance ranking
    # print(feature_importances)
    return feature_importances
#-----------------------------------------------------------


### Clear dataframe from irrelevant columns
#-----------------------------------------------------------
# Separate the features (X) and target variable (y)
def remove_unimportant(df, feature_importances):
    # Set the importance threshold (e.g., 5% of the maximum importance)
    importance_threshold = 0.05 * feature_importances['Importance'].max()

    # Filter the dataframe based on the importance threshold
    filtered_features = feature_importances.loc[feature_importances['Importance'] >= importance_threshold]

    # Print the filtered feature importance ranking
    # print(filtered_features)
    df_new  = df[filtered_features['Feature'].values]
    return df_new 
#-----------------------------------------------------------