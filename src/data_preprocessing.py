import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

            
def pre_process(df):
    # Convert Weight column to float
    df["Weight"] = df["Weight"].astype(str)
    df["Weight"] = df["Weight"].str.replace("kg", "", regex=False).str.strip().astype(float)

    # One-hot encoding
    categorical_cols = ["Company", "TypeName", "ScreenResolution", 
                        "Cpu", "Ram", "Memory", "Gpu", "OpSys"]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    def whisker(col):
        Q1,Q3 = np.percentile(col,[25,75])
        iqr = Q3 - Q1
        lw = Q1 - (1.5 * iqr)
        uw = Q3 + (1.5 * iqr)
        return lw, uw

    for i in ['Inches', 'Weight']:
        lw,uw = whisker(df[i])
        df[i] = np.where(df[i]<lw, lw, df[i])
        df[i] = np.where(df[i]>uw, uw, df[i])
        sns.boxplot(data=df, x=i)
        plt.show()

    # Data normalization
    for col in ['Inches', 'Weight']:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
        