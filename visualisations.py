import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters


sns.set_style("darkgrid")

# Visualizing the correlation matrix
def correlation_map(c_matrix):
    plt.figure(figsize=(70, 10))
    heatmap = sns.heatmap(c_matrix, vmin=-1, vmax=1, annot=True, cmap='coolwarm')
    heatmap.set_title('Correlation between Variables', fontdict={'fontsize':5}, pad=12);
    plt.savefig('c_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()