import pandas as pd
import os
import numpy as np
def load_datasets(folder_path):
    """
        Loads all CSV files from the specified folder into a dictionary of DataFrames.
        Args:
            folder_path (str): Path to the folder containing CSV files.
        Returns:
            dict: Dictionary where keys are dataset names (file names without extension) and values are pandas DataFrames.
        """
    datasets = {}
    csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

    for file in csv_files:
        dataset_name = os.path.splitext(file)[0]
        datasets[dataset_name] = pd.read_csv(os.path.join(folder_path, file), low_memory=False)

    return datasets


def extract_adni1_rids(dfs):
    """
    Extracts the unique RIDs of ADNI-1 participants.

    Args:
        dfs (dict): Dictionary where keys are dataset names and values are pandas DataFrames.

    Returns:
        set: A set of unique RIDs belonging to ADNI-1.
    """
    adni1_rids = set()
    for name, df in dfs.items():
        if "PHASE" in df.columns and "RID" in df.columns:
            adni1_rids.update(df[df["PHASE"] == "ADNI1"]["RID"].unique())
    return adni1_rids


def filter_by_adni1_rids(dfs, adni1_rids):
    """
    Filters all datasets to keep only records from ADNI-1 participants.
    """
    filtered_dfs = {}
    participant_counts = {}

    for name, df in dfs.items():
        if "RID" in df.columns:
            print(f"🔍 Before filtering {name}: {df.shape[0]} rows")

            filtered_df = df[df["RID"].isin(adni1_rids)].copy()

            # Drop unnecessary columns to reduce memory usage
            if "UPDATE_STAMP" in filtered_df.columns:
                filtered_df.drop(columns=["UPDATE_STAMP"], inplace=True)

            # Convert categorical columns to category type to save memory
            for col in filtered_df.select_dtypes(include='object').columns:
                filtered_df[col] = filtered_df[col].astype('category')

            # Drop duplicates **before returning dataset**
            filtered_df.drop_duplicates(inplace=True)

            print(f"✅ After filtering {name}: {filtered_df.shape[0]} rows")

            filtered_dfs[name] = filtered_df
            participant_counts[name] = filtered_df["RID"].nunique()

            # Save filtered dataset
            filtered_df.to_csv(f"Filtered_Data/{name}_filtered.csv", index=False)

        else:
            print(f"⚠️ Skipping {name}: Missing RID column.")

    return filtered_dfs, pd.DataFrame.from_dict(participant_counts, orient='index', columns=['Unique RIDs'])


def merge_datasets(filtered_dfs):
    """
    Merges the filtered datasets while keeping multiple visits per RID.
    """
    print("🔍 Initializing merge process...")

    # Step 1: Start with a dataset that has visit info
    base_dataset = "DXSUM_PDXCONV_ADNIALL"  # Choose a dataset that tracks visits
    if base_dataset in filtered_dfs:
        merged_df = filtered_dfs[base_dataset].copy()
        print(f"✅ Started with {base_dataset}, {merged_df.shape[0]} rows")
    else:
        raise ValueError(f"Base dataset {base_dataset} missing from filtered datasets!")

    # Step 2: Merge time-invariant datasets (on RID only)
    time_invariant_datasets = ["PTDEMOG", "FHQ", "RECBLLOG"]
    print("\n🔹 Merging Time-Invariant Datasets:")
    for name in time_invariant_datasets:
        if name in filtered_dfs:
            df = filtered_dfs[name].copy()

            # Ensure each RID appears only once in time-invariant datasets
            df.drop_duplicates(subset=["RID"], keep="first", inplace=True)

            prev_rows = merged_df.shape[0]
            merged_df = merged_df.merge(df, on="RID", how="left", suffixes=("", f"_{name}"))
            print(f"✔ Merged {name}: Rows before={prev_rows}, Rows after={merged_df.shape[0]}")

    print(f"\n✅ After merging time-invariant datasets, merged_df has {merged_df.shape[0]} rows\n")

    # Step 3: Merge longitudinal datasets (keeping multiple visits)
    longitudinal_datasets = ["MMSE", "CDR", "FAQ", "ADASSCORES", "GDSCALE", "NPIQ", "PHYSICAL", "VS", "MEDHIST"]

    print("🔹 Merging Longitudinal Datasets:")
    for name in longitudinal_datasets:
        if name in filtered_dfs:
            df = filtered_dfs[name].copy()

            # Determine merge keys (keeping visits)
            merge_keys = ["RID"]
            if "EXAMDATE" in df.columns:
                merge_keys.append("EXAMDATE")
            elif "USERDATE" in df.columns:
                merge_keys.append("USERDATE")
            elif "VISCODE" in df.columns:
                merge_keys.append("VISCODE")

            # Ensure unique rows per RID & Visit
            df.drop_duplicates(subset=merge_keys, keep="first", inplace=True)

            prev_rows = merged_df.shape[0]
            merged_df = merged_df.merge(df, on=merge_keys, how="left", suffixes=("", f"_{name}"))
            print(f"✔ Merged {name}: Rows before={prev_rows}, Rows after={merged_df.shape[0]}")

    print(f"\n✅ Final merged dataset has {merged_df.shape[0]} rows")

    return merged_df


def assign_months_since_first_visit(df):
    """
    Computes the number of months since the first visit for each RID.

    Args:
        df (pd.DataFrame): Merged dataset containing RID and EXAMDATE.

    Returns:
        pd.DataFrame: Dataset with an additional 'Months_Since_First_Visit' column.
    """
    # Ensure we have a valid date column
    if "EXAMDATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    elif "USERDATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["USERDATE"], errors="coerce")
    else:
        raise ValueError("No valid date column (EXAMDATE or USERDATE) found!")

    # Identify the earliest visit date for each RID
    df["Earliest_DATE"] = df.groupby("RID")["DATE"].transform("min")

    # Compute the difference in months (rounded)
    df["Months_Since_First_Visit"] = ((df["DATE"] - df["Earliest_DATE"]).dt.days / 30.44).round().astype("Int64")

    # Drop temporary column
    df.drop(columns=["Earliest_DATE"], inplace=True)

    return df

def round_months(df):
    """
       Rounds 'Months_Since_First_Visit' into standardized time intervals.

       Args:
           df (pd.DataFrame): Merged dataset containing 'Months_Since_First_Visit'.

       Returns:
           pd.DataFrame: Dataset with an additional 'Rounded_Months' column.
       """

    df["Rounded_Months"] = (df["Months_Since_First_Visit"]/6).round()*6

    return df


#-------------------


# Ensure output directory exists
os.makedirs("Filtered_Data", exist_ok=True)
# Load datasets from "Data" folder
datasets = load_datasets("Data")
if datasets:
    # Extract ADNI-1 RIDs
    adni1_rids = extract_adni1_rids(datasets)
    # Filter datasets to keep only ADNI-1 participants
    filtered_datasets, summary = filter_by_adni1_rids(datasets, adni1_rids)
    print(f"Initial MMSE file row count: {len(pd.read_csv('Data/MMSE.csv'))}")  # Direct from file
    print(f"Filtered MMSE row count: {len(filtered_datasets['MMSE'])}")  # After filtering
    # merge filtered datasets
    merged_data = merge_datasets(filtered_datasets)
    merged_data = assign_months_since_first_visit(merged_data)
    merged_data = round_months(merged_data)
    merged_data.to_csv("Filtered_Data/merged_dataset.csv", index=False, encoding="utf-8-sig")
    #Display summary of participants per dataset
    print(summary)
    print(merged_data[["RID", "Months_Since_First_Visit", "Rounded_Months"]].head())  # Check output
else:
    print("No datasets loaded.")

print(merged_data, merged_data.shape)