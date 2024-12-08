import pandas as pd
import kagglehub

# Download the dataset
path = kagglehub.dataset_download("sid321axn/beijing-multisite-airquality-data-set")
print("Path to dataset files:", path)

# File names to be processed
list_filenames = [
    "PRSA_Data_Guanyuan_20130301-20170228.csv",
    "PRSA_Data_Aotizhongxin_20130301-20170228.csv",
    "PRSA_Data_Wanliu_20130301-20170228.csv",
    "PRSA_Data_Tiantan_20130301-20170228.csv",
    "PRSA_Data_Wanshouxigong_20130301-20170228.csv",
    "PRSA_Data_Nongzhanguan_20130301-20170228.csv",
    "PRSA_Data_Shunyi_20130301-20170228.csv",
    "PRSA_Data_Changping_20130301-20170228.csv",
    "PRSA_Data_Dingling_20130301-20170228.csv",
    "PRSA_Data_Huairou_20130301-20170228.csv",
    "PRSA_Data_Gucheng_20130301-20170228.csv",
    "PRSA_Data_Dongsi_20130301-20170228.csv"
]

# Load datasets into a list of dataframes
dataframes = []
for filename in list_filenames:
    full_path = f"{path}/{filename}"
    df = pd.read_csv(full_path)
    df['site'] = filename.split('_')[2]  # Add site identifier
    dataframes.append(df)

# Combine all dataframes into one
combined_data = pd.concat(dataframes, ignore_index=True)

# Display combined data structure
print("Combined Dataset Info:")
print(combined_data.info())
