import pandas as pd
import numpy as np
import os

class ADNIProcessor:
    """
    Processes ADNI datasets by loading, filtering, merging, and extracting baseline measures.
    """
    def __init__(self, data_folder="..//Data", output_folder="..//Filtered_Data"):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.datasets = {}
        self.filtered_datasets = {}
        self.adni1_rids = set()
        self.merged_data = None
        os.makedirs(self.output_folder, exist_ok=True)

    def load_datasets(self):
        """
        Load all CSV files from the specified folder into a dictionary of DataFrames.
        """
        print("** Loading Datasets... **")
        csv_files = [file for file in os.listdir(self.data_folder) if file.endswith(".csv")]
        for file in csv_files:
            dataset_name = os.path.splitext(file)[0]
            self.datasets[dataset_name] = pd.read_csv(os.path.join(self.data_folder, file), low_memory=False)
            print(f"** {file} loaded. **")

    def extract_adni1_rids(self):
        """
        Extracts unique ADNI-1 participant RIDs from datasets containing 'Phase' and 'RID'.
        """
        print("** Extracting ADNI-1 RIDs... **")
        for df in self.datasets.values():
            if "Phase" in df.columns and "RID" in df.columns:
                self.adni1_rids.update(df[df['Phase'] == "ADNI1"]["RID"].unique())
        print(f"** Found {len(self.adni1_rids)} unique ADNI-1 RIDS. **")

    def filter_by_adni1_rids(self):
        """
        Filters each dataset to retain only ADNI-1 participants.
        Converts object columns to category, removes duplicates,
        and saves each filtered dataset.
        Returns a summary DataFrame with unique RID counts per dataset.
        """
        print("** Filtering datasets for ADNI-1 participants. **")
        participant_counts = {}
        for name, df in self.datasets.items():
            if "RID" not in df.columns:
                print(f"** Skipping {name}: Missing RID column. **")
                continue
            filtered = df[df["RID"].isin(self.adni1_rids)].copy()
            if "UPDATE_STAMP" in filtered.columns:
                filtered.drop(columns=["UPDATE_STAMP"], inplace=True)
            for col in filtered.select_dtypes(include="object").columns:
                filtered[col] = filtered[col].astype("category")
            filtered.drop_duplicates(inplace=True)
            self.filtered_datasets[name] = filtered
            participant_counts[name] = filtered["RID"].nunique()
            filtered.to_csv(f"{self.output_folder}/{name}_filtered.csv", index=False)
        return pd.DataFrame.from_dict(participant_counts, orient="index", columns=["Unique RIDs"])

    def merge_datasets(self):
        """
        Merges filtered datasets into a single DataFrame using DXSUM_PDXCONV_ADNIALL as the base.
        Non-key columns from each merging dataset are renamed with a suffix indicating their source.
        """
        print("** Merging datasets... **")
        base = "DXSUM_PDXCONV_ADNIALL"
        if base not in self.filtered_datasets:
            raise ValueError(f"Base dataset {base} missing!")
        self.merged_data = self.filtered_datasets[base].copy()
        print(f"** Started with {base}, {self.merged_data.shape[0]} rows. **")

        # Merge time-invariant datasets
        time_invariant_datasets = ["PTDEMOG", "FHQ", "RECBLLOG"]
        print("** Merging Time-Invariant Datasets **")
        for ds in time_invariant_datasets:
            if ds in self.filtered_datasets:
                tmp = self.filtered_datasets[ds].drop_duplicates(subset=["RID"]).copy()
                merge_keys = ["RID"]
                # Rename every column except the merge keys with a suffix
                rename_dict = {col: f"{col}_{ds}" for col in tmp.columns if col not in merge_keys}
                tmp.rename(columns=rename_dict, inplace=True)
                prev_rows = self.merged_data.shape[0]
                self.merged_data = self.merged_data.merge(tmp, on="RID", how="left")
                print(f"** Merged {ds}: Rows before={prev_rows}, after={self.merged_data.shape[0]}. **")

        # Merge longitudinal datasets
        longitudinal_datasets = ["MMSE", "CDR", "FAQ", "ADASSCORES", "GDSCALE", "NPIQ", "PHYSICAL", "VS", "MEDHIST"]
        print("** Merging Longitudinal Datasets **")
        for ds in longitudinal_datasets:
            if ds in self.filtered_datasets:
                tmp = self.filtered_datasets[ds].copy()
                merge_keys = ["RID"]
                # Choose a merge key among date columns if available
                for key in ["EXAMDATE", "USERDATE", "VISCODE"]:
                    if key in tmp.columns:
                        merge_keys.append(key)
                        break
                tmp.drop_duplicates(subset=merge_keys, inplace=True)
                # Rename non-key columns with the dataset-specific suffix
                rename_dict = {col: f"{col}_{ds}" for col in tmp.columns if col not in merge_keys}
                tmp.rename(columns=rename_dict, inplace=True)
                prev_rows = self.merged_data.shape[0]
                self.merged_data = self.merged_data.merge(tmp, on=merge_keys, how="left")
                print(f"** Merged {ds}: Rows before={prev_rows}, after={self.merged_data.shape[0]}. **")
        print(f"** Final merged dataset has {self.merged_data.shape[0]} rows. **")

    def assign_months_since_first_visit(self):
        """
         Computes the number of months since each participant's first visit using EXAMDATE or USERDATE.
        """
        date_col = "EXAMDATE" if "EXAMDATE" in self.merged_data.columns else "USERDATE"
        self.merged_data["DATE"] = pd.to_datetime(self.merged_data[date_col], errors="coerce")
        self.merged_data["Earliest_DATE"] = self.merged_data.groupby("RID")["DATE"].transform("min")
        self.merged_data["Months_Since_First_Visit"] = (
                (self.merged_data["DATE"] - self.merged_data["Earliest_DATE"]).dt.days / 30.44
        ).round().astype("Int64")
        self.merged_data.drop(columns=["Earliest_DATE"], inplace=True)

    def round_months(self):
        """
        Rounds 'Months_Since_First_Visit' into standardized time intervals.
        """
        self.merged_data["Rounded_Months"] = (self.merged_data["Months_Since_First_Visit"] / 6).round() * 6

    def delete_cols(self):
        """
        Drops columns with high missingness (>82.3%) and those matching unwanted patterns.
        """
        missing = 100 * self.merged_data.isnull().sum() / len(self.merged_data)
        drop_cols = missing[missing == 100].index.tolist()
        patterns = ["userdate", "update_stamp", "viscode", "siteid", "phase", "id_"]
        for pattern in patterns:
            drop_cols += [col for col in self.merged_data.columns if pattern in col.lower()]
        drop_cols += ["DXDEP", "DXCHANGE", "DXPARKSP"]
        self.merged_data.drop(columns=list(set(drop_cols)), inplace=True)

    def filter_0_36_months(self):
        """
        Filters the merged data to include only visits between 0 and 36 months.
        """
        initial_count = self.merged_data.shape[0]
        self.merged_data = self.merged_data[self.merged_data["Months_Since_First_Visit"].between(0, 36)]
        print(f"** Filtered data to 0-36 months: {initial_count} -> {self.merged_data.shape[0]} rows. **")
    def extract_baseline_vars(self):
        """
        Extracts baseline values (first non-missing based on Months_Since_First_Visit) for variables
        ending with specified suffixes and saves each as a CSV.
        """
        # Sort the merged data by RID and visit time
        df = self.merged_data.sort_values(by=["RID", "Months_Since_First_Visit"])

        def get_baseline(group, suffix):
            baseline = {}
            # Select columns ending with the specified suffix
            cols = [col for col in group.columns if col.endswith(suffix)]
            for col in cols:
                valid = group[group[col].notnull() & group["Months_Since_First_Visit"].notnull()]
                if not valid.empty:
                    idx = valid["Months_Since_First_Visit"].idxmin()
                    baseline[col] = valid.loc[idx, col]
                else:
                    baseline[col] = np.nan
            baseline["RID"] = group.iloc[0]["RID"]
            valid_months = group["Months_Since_First_Visit"].dropna()
            baseline["Baseline_Month"] = valid_months.min() if not valid_months.empty else np.nan
            return pd.Series(baseline)

        # Define suffixes and their output filenames
        suffixes = {"_GDSCALE": "baseline_GDSCALE.csv",
                    "_MMSE": "baseline_MMSE.csv",
                    "_CDR": "baseline_CDR.csv",
                    "_NPIQ": "baseline_NPIQ.csv",
                    "FAQ": "baseline_FAQ.csv",
                    "_ADASSCORES": "baseline_ADASSCORES.csv"}

        for suffix, filename in suffixes.items():
            baseline_df = df.groupby("RID").apply(lambda g: get_baseline(g, suffix)).reset_index(drop=True)
            baseline_df.to_csv(f"{self.output_folder}/{filename}", index=False)
            print(f"Baseline dataset for variables ending with '{suffix}' saved in {self.output_folder}/{filename}")

    def impute_missing_values(self):
        """
        Imputes missing values in self.merged_data.
        - Numeric columns: fill missing values with the column median (if non-empty).
        - Non-numeric columns: fill missing values with the column mode (if available).

        This is imputing missing values only for columns that have missing values < 0.5%
        """
        # Impute numeric columns
        numeric_cols = self.merged_data.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if self.merged_data[col].notnull().sum() > 0:
                median_value = self.merged_data[col].median()
                self.merged_data[col] = self.merged_data[col].fillna(median_value)
            else:
                # If the column is entirely missing, you can choose to leave it as is or fill with a default value.
                self.merged_data[col] = self.merged_data[col].fillna(np.nan)

        # Impute non-numeric columns
        non_numeric_cols = self.merged_data.select_dtypes(exclude=["number"]).columns
        for col in non_numeric_cols:
            if not self.merged_data[col].mode().empty:
                mode_value = self.merged_data[col].mode()[0]
                self.merged_data[col] = self.merged_data[col].fillna(mode_value)

    def filter_baseline_data(self, method="exact_zero"):
        """
        Filters the merged dataset to include only baseline rows.

        Parameters:
            method (str): Determines the definition of baseline:
                - "exact_zero": Retain rows where Months_Since_First_Visit == 0.
                - "earliest": Retain the earliest visit for each participant.

        The filtered baseline dataset is saved as 'baseline_only_dataset.csv' in the output folder.
        """
        if self.merged_data is None:
            raise ValueError("Merged dataset is not available!")

        if method == "exact_zero":
            baseline_df = self.merged_data[self.merged_data["Months_Since_First_Visit"] == 0].copy()
        elif method == "earliest":
            # Sort by participant (RID) and visit time, then group and select the first record.
            df_sorted = self.merged_data.sort_values(by=["RID", "Months_Since_First_Visit"])
            baseline_df = df_sorted.groupby("RID", as_index=False).first()
        else:
            raise ValueError("Invalid method. Use 'exact_zero' or 'earliest'.")

        print(f"Baseline dataset has {baseline_df.shape[0]} rows.")
        baseline_df.to_csv(f"{self.output_folder}/baseline_only_dataset.csv", index=False)

    def save_final_dataset(self):
        """
        Saves the final processed dataset.
        """
        self.merged_data.to_csv(f"{self.output_folder}/merged_dataset.csv", index=False, encoding="utf-8-sig")
        print(f"** Final dataset saved to {self.output_folder}/merged_dataset.csv **")

    def run_pipeline(self):
        """Executes the full processing pipeline."""
        self.load_datasets()
        self.extract_adni1_rids()
        summary = self.filter_by_adni1_rids()
        self.merge_datasets()
        self.assign_months_since_first_visit()
        self.round_months()
        self.filter_0_36_months()
        self.impute_missing_values()
        self.delete_cols()
        self.save_final_dataset()
        print(summary)


