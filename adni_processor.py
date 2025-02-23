import pandas as pd
import numpy as np
import os

class ADNIProcessor:
    """
    A class to handle loading, filtering, and processing of ADNI datasets.
    """
    def __init__(self, data_folder="Data", output_folder="Filtered_Data"):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.datasets = {}
        self.filtered_datasets = {}
        self.adni1_rids = set()
        self.merged_data = None
        # Ensure output directory exists
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
        Extracts the unique RIDs of ADNI1-participants.
        """
        print("** Extracting ADNI-1 RIDs... **")
        for name, df in self.datasets.items():
            if "Phase" in df.columns and "RID" in df.columns:
                self.adni1_rids.update(df[df['Phase'] == "ADNI1"]["RID"].unique())
        print(f"** Found {len(self.adni1_rids)} unique ADNI-1 RIDS. **")

    def filter_by_adni1_rids(self):
        """
        Filters all datasets to keep only records from ADNI-1 participants.
        """
        print(f"** Filtering datasets for ADNI-1 participants. **")
        participant_counts = {}

        for name, df in self.datasets.items():
            if "RID" in df.columns:
                filtered_df = df[df["RID"].isin(self.adni1_rids)].copy()
                # Drop unnecessary columns
                if "UPDATE_STAMP" in filtered_df.columns:
                    filtered_df.drop(columns=["UPDATE_STAMP"], inplace=True)

                    # Convert categorical columns
                for col in filtered_df.select_dtypes(include='object').columns:
                    filtered_df[col] = filtered_df[col].astype('category')
                # Remove duplicates
                filtered_df.drop_duplicates(inplace=True)

                self.filtered_datasets[name] = filtered_df
                participant_counts[name] = filtered_df["RID"].nunique()

                # Save filtered dataset
                filtered_df.to_csv(f"{self.output_folder}/{name}_filtered.csv", index=False)
        else:
            print(f"** Skipping {name}: Missing RID column. **")

        return pd.DataFrame.from_dict(participant_counts, orient='index', columns=['Unique RIDs'])

    def merge_datasets(self):
        """
        Merges the filtered datasets while keeping multiple visits per RID.
        """
        print("** Merging datasets... **")
        base_dataset = "DXSUM_PDXCONV_ADNIALL"
        if base_dataset in self.filtered_datasets:
            self.merged_data = self.filtered_datasets[base_dataset].copy()
            print(f"** Started with {base_dataset}, {self.merged_data.shape[0]} rows. **")
        else:
            raise ValueError(f"Base dataset {base_dataset} missing from filtered datasets!")

        # Merge time-invariant datasets
        time_invariant_datasets = ["PTDEMOG", "FHQ", "RECBLLOG"]
        print("** Merging Time-Invariant Datasets **")
        for name in time_invariant_datasets:
            if name in self.filtered_datasets:
                df = self.filtered_datasets[name].copy()
                df.drop_duplicates(subset=["RID"], keep="first", inplace=True)
                prev_rows = self.merged_data.shape[0]
                self.merged_data = self.merged_data.merge(df, on="RID", how="left", suffixes=("", f"_{name}"))
                print(f"** Merged {name}: Rows before={prev_rows}, Rows after={self.merged_data.shape[0]}. **")

        # Merge longitudinal datasets
        longitudinal_datasets = ["MMSE", "CDR", "FAQ", "ADASSCORES", "GDSCALE", "NPIQ", "PHYSICAL", "VS", "MEDHIST"]
        print("** Merging Longitudinal Datasets **")
        for name in longitudinal_datasets:
            if name in self.filtered_datasets:
                df = self.filtered_datasets[name].copy()
                merge_keys = ["RID"]

                if "EXAMDATE" in df.columns:
                    merge_keys.append("EXAMDATE")
                elif "USERDATE" in df.columns:
                    merge_keys.append("USERDATE")
                elif "VISCODE" in df.columns:
                    merge_keys.append("VISCODE")

                df.drop_duplicates(subset=merge_keys, keep="first", inplace=True)
                prev_rows = self.merged_data.shape[0]
                self.merged_data = self.merged_data.merge(df, on=merge_keys, how="left", suffixes=("", f"_{name}"))
                print(f"** Merged {name}: Rows before={prev_rows}, Rows after={self.merged_data.shape[0]}. **")

            print(f"** Final merged dataset has {self.merged_data.shape[0]} rows. **")

    def assign_months_since_first_visit(self):
        """
        Computes the number of months since the first visit for each RID.
        """
        if "EXAMDATE" in self.merged_data.columns:
            self.merged_data["DATE"] = pd.to_datetime(self.merged_data["EXAMDATE"], errors="coerce")
        elif "USERDATE" in self.merged_data.columns:
            self.merged_data["DATE"] = pd.to_datetime(self.merged_data["USERDATE"], errors="coerce")
        else:
            raise ValueError("No valid date column (EXAMDATE or USERDATE) found!")

        self.merged_data["Earliest_DATE"] = self.merged_data.groupby("RID")["DATE"].transform("min")
        self.merged_data["Months_Since_First_Visit"] = (
                    (self.merged_data["DATE"] - self.merged_data["Earliest_DATE"]).dt.days / 30.44).round().astype(
            "Int64")
        self.merged_data.drop(columns=["Earliest_DATE"], inplace=True)

    def round_months(self):
        """
        Rounds 'Months_Since_First_Visit' into standardized time intervals.
        """
        self.merged_data["Rounded_Months"] = (self.merged_data["Months_Since_First_Visit"] / 6).round() * 6

    def save_final_dataset(self):
        """
        Saves the final processed dataset.
        """
        self.merged_data.to_csv(f"{self.output_folder}/merged_dataset.csv", index=False, encoding="utf-8-sig")
        print(f"** Final dataset saved to {self.output_folder}/merged_dataset.csv **")

    def run_pipeline(self):
        self.load_datasets()
        self.extract_adni1_rids()
        summary = self.filter_by_adni1_rids()

        self.merge_datasets()
        self.assign_months_since_first_visit()
        self.round_months()
        self.save_final_dataset()
        print(summary)



