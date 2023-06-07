import numpy as np
import pandas as pd


import visualisations as vi
import analysis as an


df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
# df.info()
# df.head()
# df.duplicated().sum()
# df.isna().sum()


### Convert categorical string variable values into numeric codes :
#-----------------------------------------------------------
# Get categorical variable columns unique values series
df1 = df.loc[:, df.dtypes == 'object']
string_value_list=pd.Series({c: df1[c].unique() for c in df1})
cat_cols = string_value_list.keys()
# print(string_value_list)

# Final Calculation for numeric conversion 
for col, uni_val in string_value_list.items():
    map_dict = {k: v for v, k in enumerate(uni_val)}
    df[col]= df[col].map(map_dict)
#-----------------------------------------------------------


## ANOVA analysis for defining variable irrelevancy with the target
# anova_analysis(cat_cols, df['Attrition'])


# Remove columns below a specified importance degree
fi = an.fi_analysis(df, 'Attrition')
df_imp = an.remove_unimportant(df, fi)

# Correlation visualization between 'Attrition' and other variables
vi.correlation_map(df, 'Attrition')
