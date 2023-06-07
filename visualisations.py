import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters


sns.set_style("darkgrid")

# Visualizing the correlation matrix
def correlation_map(df, target):
    cor = df.corr()[[target]].sort_values(by=target, ascending=False)
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(cor, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Correlation Heatmap for Attrition',
                    fontdict={'fontsize': 18}, pad=12)
    plt.show()