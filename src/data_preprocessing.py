import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import re

          
class preprocess:         
    @staticmethod
    def pre_process(df):
        df["Weight"] = df["Weight"].astype(str)
        df["Weight"] = df["Weight"].str.replace("kg", "", regex=False).str.strip().astype(float)
        df["Ram"] = df["Ram"].str.replace("GB", "", regex=False).str.strip().astype(float)
        
        # Extract company (first word)
        df["cpu_company"] = df["Cpu"].str.split().str[0]

        # Extract CPU speed (number before GHz)
        df["cpu_speed"] = df["Cpu"].str.extract(r'(\d+\.?\d*)GHz')[0].astype(float)
        df.head()

        def extract_storage(row, storage_type):
            matches = re.findall(r'([\d.]+)\s*(TB|GB)\s*([A-Za-z ]+)', row)
            total = 0
            for size, unit, stype in matches:
                stype = stype.strip().lower()
                value = float(size)
                # convert GB -> TB
                if unit.upper() == 'GB':
                    value /= 1024
                if storage_type in stype:
                    total += value
            return total

        # Create 4 new columns
        for stype in ["Hybrid", "SSD", "HDD", "FlashStorage"]:
            df[stype] = df["Memory"].apply(lambda x: extract_storage(x, stype.lower()))

        df["Resolution_Height"] = df["ScreenResolution"].str.extract(r'x(\d+)').astype(float)
        df["FlashStorage"] = df["FlashStorage"].astype(float)

        df.drop(columns=["ScreenResolution", "Cpu", "Memory"], inplace=True)   

        numeric_cols = ["Inches", "Ram", "Weight", "cpu_speed", "Resolution_Height", "SSD", "HDD"]

        # Handle outliers using IQR method
        def whisker(col):
            Q1,Q3 = np.percentile(col,[25,75])
            iqr = Q3 - Q1
            lw = Q1 - (1.5 * iqr)
            uw = Q3 + (1.5 * iqr)
            return lw, uw

        for i in numeric_cols:
            lw,uw = whisker(df[i])
            df[i] = np.where(df[i]<lw, lw, df[i])
            df[i] = np.where(df[i]>uw, uw, df[i])
    
        # Z-score standardization
        number = df.select_dtypes(include=np.number).columns.tolist()
        number.remove("Price")

        for col in number:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
    
        # One-hot encoding
        categorical_cols = ["Company", "TypeName", "OpSys", "cpu_company", "Gpu"]

        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)

        output_path = "/Users/avi/Desktop/university/sem_5/Machine_Learning/Avi_Dhall_A1/laptop_price_task-/data/clean.csv"
        df.to_csv(output_path, index=False)

# Load the CSV into a DataFrame
# file_path = "/Users/avi/Desktop/university/sem_5/Machine_Learning/Avi_Dhall_A1/laptop_price_task-/data/train_data.csv"
# df = pd.read_csv(file_path)

# Call the preprocessing method
# preprocessor = preprocess()
# preprocessor.pre_process(df)
