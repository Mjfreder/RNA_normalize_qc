import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.robust.norms import HuberT
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.stats.multitest import multipletests


# Load the combined RNA-seq data
# Load the combined RNA-seq data and save as Pickle for faster access
def load_combined_data(file_path, pickle_file="combined_data.pkl"):
    if os.path.exists(pickle_file):
        print(f"[LOG] Loading combined RNA-seq data from Pickle file: {pickle_file}")
        data = pd.read_pickle(pickle_file)
    else:
        print(f"[LOG] Loading combined RNA-seq data from CSV file: {file_path}")
        data = pd.read_csv(file_path)
        print(f"[LOG] Saving combined RNA-seq data as Pickle: {pickle_file}")
        data.to_pickle(pickle_file)  # Save the dataframe as Pickle for faster reloading
    return data


# Load the gene reference file
def load_gene_reference(file_path):
    gene_reference = pd.read_csv(file_path)
    print(f"[LOG] Loaded gene reference data from {file_path}")
    return gene_reference


# Step 1: Filter out non-protein-coding genes and create output with additional "count" column
def filter_protein_coding_genes(data, gene_reference):
    # Merge data with the gene reference to filter out non-protein-coding genes
    filtered_data = pd.merge(data, gene_reference, how="inner", on="gene_id")
    print("[LOG] Filtered data to include only protein-coding genes")

    # Add 'count' column as the first column (for demonstration, using row numbers as counts)
    filtered_data.insert(0, "count", range(1, len(filtered_data) + 1))

    # Save the step 1 output
    filtered_data.to_csv("step1.csv", index=False)
    print("[LOG] Saved step 1 output to 'step1.csv'")

    return filtered_data


# Step 2: Calculate the 75th percentile of counts for each sample and create step2_MUQ.csv
def calculate_75th_percentile(filtered_data):
    percentiles = {}

    # Identify expected_count columns (every third column starting from 4th column)
    sample_columns = filtered_data.columns[4::3]

    for sample in sample_columns:
        # Ensure that the column data is numeric
        sample_data = pd.to_numeric(filtered_data[sample], errors="coerce")
        sample_data = sample_data.replace(0, np.nan)  # Exclude zero counts

        # Check if the column has valid numeric data
        if sample_data.notna().sum() > 0:
            s_75 = sample_data.quantile(0.75)
            percentiles[sample] = s_75
            print(f"[LOG] Calculated 75th percentile for {sample}: {s_75}")
        else:
            percentiles[sample] = np.nan
            print(f"[LOG] No valid data for {sample}, setting 75th percentile as NaN.")

    # Calculate the Median Upper Quartile (MUQ)
    valid_percentiles = [v for v in percentiles.values() if not np.isnan(v)]
    muq = np.median(valid_percentiles) if valid_percentiles else np.nan

    # Extract sample names (corresponding to the samples like "SP17-2267-FSA10")
    sample_names = [
        filtered_data.columns[i - 2] for i in range(4, len(filtered_data.columns), 3)
    ]

    # Create the step2_MUQ.csv output
    step2_data = pd.DataFrame(
        {
            "Sample": sample_names,
            "Count 75th PCTLS": [percentiles[sample] for sample in sample_columns],
            "MUQ": [muq] + [""] * (len(sample_columns) - 1),
        }
    )
    step2_data.to_csv("step2_MUQ.csv", index=False)
    print("[LOG] Saved step 2 output to 'step2_MUQ.csv'")

    return percentiles, muq


# Step 3: Apply gene size transformation (max(x, 252)) and calculate the median gene sizes
def apply_gene_size_transformation(data):
    # Identify effective_length columns (every third column starting from the third column)
    length_columns = data.columns[3::3]

    # Apply max(effective_length, 252) to each effective length column
    for length_column in length_columns:
        data[length_column] = data[length_column].apply(lambda x: max(x, 252))
        print(f"[LOG] Applied max(effective_length, 252) to {length_column}")

    # Collect all transformed gene sizes
    all_lengths = []
    for length_column in length_columns:
        all_lengths.extend(data[length_column].tolist())

    overall_median = np.median(all_lengths)

    # Save the transformed data to step3_MGS.csv
    data.to_csv("step3_MGS.csv", index=False)
    print("[LOG] Saved step 3 output to 'step3_MGS.csv'")
    print(f"[LOG] Global Median Gene Size (MGS): {overall_median}")

    return data, overall_median


# Step 4: Normalize FPKM-UQ using vectorized operations
def normalize_fpkm_uq(
    data, percentiles, muq, global_mgs, file_name="step4_fpkm_uq.csv"
):
    normalized_data = data.copy()
    sample_columns = normalized_data.columns[4::3]  # Sample columns

    # Vectorized normalization: (reads / (gene_length * s75th)) * muq * global_mgs
    for i, sample in enumerate(sample_columns):
        s75th = percentiles[sample]
        gene_lengths = normalized_data.iloc[:, 3::3].values[
            :, i
        ]  # Vector of gene lengths
        reads = normalized_data[sample].values

        # Vectorized calculation of normalized values
        normalized_values = (reads / (gene_lengths * s75th)) * muq * global_mgs
        normalized_data[sample] = normalized_values

    normalized_data.to_csv(file_name, index=False)
    print(f"[LOG] Saved normalized FPKM-UQ data to '{file_name}'")
    return normalized_data


# Step 5: Log2 transformation of FPKM-UQ
# Step 5: Log2 transformation using vectorized operations
def log2_transform_fpkm_uq(data, file_name="step5_log2_fpkm_uq.csv"):
    log2_data = data.copy()
    fpkm_uq_columns = log2_data.columns[4::3]  # Sample columns

    # Vectorized log2 transformation for all sample columns at once
    log2_data[fpkm_uq_columns] = np.log2(log2_data[fpkm_uq_columns] + 0.01)

    log2_data.to_csv(file_name, index=False)
    print(f"[LOG] Saved log2 transformed FPKM-UQ data to '{file_name}'")
    return log2_data


# Step 6: Calculate MAD and identify outliers
def calculate_mad_and_identify_outliers(data, file_name="step6_mad_outliers.csv"):
    mad_data = data.copy()
    sample_columns = data.columns[4::3]

    mad_data["cohort_median"] = mad_data[sample_columns].median(axis=1)
    print("[LOG] Calculated cohort median for each gene")

    for sample in sample_columns:
        mad_column_name = f"MAD_{sample}"
        mad_data[mad_column_name] = (mad_data[sample] - mad_data["cohort_median"]).abs()
        print(f"[LOG] Calculated MAD values for {sample}")

    average_mad_values = {
        sample: mad_data[f"MAD_{sample}"].mean() for sample in sample_columns
    }

    output_columns = [f"MAD_{sample}" for sample in sample_columns] + ["cohort_median"]
    output_data = mad_data[output_columns]
    output_data.to_csv(file_name, index=False)
    print(f"[LOG] Saved MAD and cohort median data to '{file_name}'")

    return mad_data, average_mad_values


# Step 7-8: Apply FDR method to identify outliers and save results
def apply_fdr_method(average_mad_values, alpha=0.01):
    average_mad_df = pd.DataFrame(
        list(average_mad_values.items()), columns=["Sample", "Average MAD"]
    )

    rlm_model = RLM(
        average_mad_df["Average MAD"],
        np.ones_like(average_mad_df["Average MAD"]),
        M=HuberT(),
    )
    rlm_results = rlm_model.fit()

    residuals = average_mad_df["Average MAD"] - rlm_results.fittedvalues
    pvals = 2 * (1 - norm.cdf(residuals.abs() / rlm_results.scale))

    _, corrected_pvals, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    average_mad_df["Raw P-Value"] = pvals
    average_mad_df["Adjusted P-Value"] = corrected_pvals
    average_mad_df["Residuals"] = residuals
    average_mad_df["Outlier"] = corrected_pvals < alpha
    average_mad_df["Outlier"] = average_mad_df["Outlier"].apply(
        lambda x: "Yes" if x else "No"
    )

    print(f"[LOG] Applied FDR method with alpha={alpha}")
    return average_mad_df




def save_average_mad_with_outliers(average_mad_df, file_name="step8_outliers.csv"):
    columns_to_save = [
        "Sample",
        "Average MAD",
        "Residuals",
        "Raw P-Value",
        "Adjusted P-Value",
        "Outlier",
    ]
    average_mad_df.to_csv(file_name, columns=columns_to_save, index=False)
    print(f"[LOG] Saved average MAD values with outlier status and additional metrics to '{file_name}'")



# Step 9-14: Rescale cohorts to have a global median of 7, adjust log2 values, and handle outliers
def rescale_cohorts(data, outlier_samples):
    sample_columns = data.columns[4::3]

    cohort_medians = data[sample_columns].apply(
        lambda row: np.median(
            [
                val
                for i, val in enumerate(row)
                if sample_columns[i] not in outlier_samples
            ]
        ),
        axis=1,
    )

    data["Cohort Median Excl. Outliers"] = cohort_medians
    filtered_medians = cohort_medians[cohort_medians > -6.6]
    global_median = np.median(filtered_medians)
    delta = 7 - global_median

    data["Adjusted Cohort Median"] = np.where(
        cohort_medians > -6.6, cohort_medians + delta, cohort_medians
    )
    return data, delta


def adjust_log2_fpkm_uq(data, delta, file_name="step14_adj_log2_fpkm_uq.csv"):
    adjusted_data = data.copy()
    sample_columns = data.columns[4::3]

    for column in sample_columns:
        adjusted_data[column] = np.where(
            adjusted_data[column] > -6.6,
            adjusted_data[column] + delta,
            adjusted_data[column],
        )
        print(f"[LOG] Applied delta adjustment to {column}")

    adjusted_data.to_csv(file_name, index=False)
    print(f"[LOG] Saved adjusted Log2 FPKM-UQ data to '{file_name}'")
    return adjusted_data


# Step 15: Adjust log2 values for all samples
def adjust_all_samples(data, delta, file_name="step15_final_adjusted_log2_fpkm_uq.csv"):
    final_adjusted_data = adjust_log2_fpkm_uq(data, delta)
    final_adjusted_data.to_csv(file_name, index=False)
    print(f"[LOG] Saved final adjusted Log2 FPKM-UQ data to '{file_name}'")
    return final_adjusted_data


# Step 15a: Replace negative log2 values with zero
def replace_negative_values_with_zero(
    data, file_name="step15a_replace_negative_with_zero.csv"
):
    adjusted_data = data.copy()
    sample_columns = data.columns[4::3]

    for column in sample_columns:
        adjusted_data[column] = adjusted_data[column].apply(lambda x: max(x, 0))
        print(f"[LOG] Replaced negative values with zero in {column}")

    adjusted_data.to_csv(file_name, index=False)
    print(f"[LOG] Saved data with negative values replaced with zero to '{file_name}'")
    return adjusted_data


# Step 16: Validate re-scaling global median and document adjustments
def validate_rescaling(
    data, outlier_samples, initial_global_median, file_name="rescaling_validation.txt"
):
    rescaled_data, new_delta = rescale_cohorts(data, outlier_samples)
    new_global_median = np.median(rescaled_data["Adjusted Cohort Median"])

    with open(file_name, "w") as f:
        f.write(f"Initial Global Median: {initial_global_median}\n")
        f.write(f"New Adjusted Global Median: {new_global_median}\n")

    print(f"[LOG] Documented global median rescaling in '{file_name}'")
    return rescaled_data, new_global_median


# Step 17: Save final output
def save_final_output(data, file_name="final_adjusted_log2_fpkm_uq.csv"):
    # List of columns to keep
    output_columns = ["gene_id", "Symbol"] + list(data.columns[4::3])

    # Ensure "Adjusted Cohort Median" is not included
    output_columns = [col for col in output_columns if col in data.columns and col != "Adjusted Cohort Median"]

    # Save the final output without "Adjusted Cohort Median"
    final_output_data = data[output_columns]
    final_output_data.to_csv(file_name, index=False)

    print(f"[LOG] Saved final output to '{file_name}' (excluding 'Adjusted Cohort Median')")
    return final_output_data




def load_sample_id_mapping(combined_data_file):
    """Extracts all Sample IDs from row 2 of 'combined_data.csv' under headers 'Sample_ID', 'Sample_ID.1', etc.'"""
    
    # Read first two rows (headers and Sample ID row)
    combined_data = pd.read_csv(combined_data_file, nrows=2)

    # Debugging: Print the actual headers
    print(f"[DEBUG] Column Headers in combined_data.csv: {list(combined_data.columns)}")

    # Identify **all** columns where the header starts with "Sample_ID"
    sample_id_mapping = {}
    
    for col in combined_data.columns:
        if col.startswith("Sample_ID"):  # Identify all Sample_ID columns
            sample_id_mapping[col] = combined_data.iloc[1][col]  # Extract Sample ID from row 2

    print(f"[LOG] Sample ID Mapping Extracted: {sample_id_mapping}")  # Debugging log
    return sample_id_mapping






# Function to rename columns in final output files
import re

def rename_sample_rows(file_name, sample_id_mapping):
    """Renames the 'Sample' column values in step8_outliers.csv using the correct Sample ID without '_FPKM_UQ'."""
    
    if not os.path.exists(file_name):
        print(f"[LOG] File '{file_name}' not found, skipping renaming.")
        return

    df = pd.read_csv(file_name)

    # Debugging: Print current values before renaming
    print(f"[DEBUG] Sample column values before renaming in {file_name}: {df['Sample'].tolist()}")

    # Convert mapping to a list for sequential replacement
    sample_id_list = list(sample_id_mapping.values())
    sample_id_index = 0  # Track correct Sample ID assignment

    # Replace values in the 'Sample' column
    for i in range(len(df)):
        if df.loc[i, "Sample"].startswith("expected_count"):
            if sample_id_index < len(sample_id_list):  # Ensure we don't go out of bounds
                df.loc[i, "Sample"] = sample_id_list[sample_id_index]  # No '_FPKM_UQ'
                sample_id_index += 1  # Move to the next Sample ID

    # Debugging: Print updated column values
    print(f"[DEBUG] Sample column values after renaming in {file_name}: {df['Sample'].tolist()}")

    # Save the updated file
    df.to_csv(file_name, index=False)

    print(f"[LOG] Successfully updated 'Sample' column names in '{file_name}'")

def rename_sample_columns(file_name, sample_id_mapping):
    """Renames 'expected_count.X' columns using the correct Sample ID and appends '_FPKM_UQ'."""
    
    if not os.path.exists(file_name):
        print(f"[LOG] File '{file_name}' not found, skipping renaming.")
        return

    df = pd.read_csv(file_name)

    # Debugging: Print current headers before renaming
    print(f"[DEBUG] Headers before renaming in {file_name}: {df.columns.tolist()}")

    # Dictionary to store new column names
    new_column_names = {}

    # Convert sample_id_mapping into a list for sequential replacement
    sample_id_list = list(sample_id_mapping.values())
    sample_id_index = 0  # Track correct Sample ID assignment

    for col in df.columns:
        if col.startswith("expected_count"):
            if sample_id_index < len(sample_id_list):  # Ensure we don't go out of bounds
                new_column_names[col] = f"{sample_id_list[sample_id_index]}_FPKM_UQ"
                sample_id_index += 1  # Move to the next Sample ID
            else:
                new_column_names[col] = col  # Keep unchanged if no available Sample ID
        else:
            new_column_names[col] = col  # Keep non-'expected_count' columns unchanged

    # Debugging: Print new column mappings
    print(f"[DEBUG] Column Renaming Mapping: {new_column_names}")

    # Rename and overwrite file
    df.rename(columns=new_column_names, inplace=True)
    df.to_csv(file_name, index=False)

    print(f"[LOG] Successfully updated column names in '{file_name}'")


# Step18 convert log2 to linear space
def convert_log2_to_linear(input_file, output_file):
    """Converts log2-transformed FPKM-UQ values back to linear scale, retaining the first two columns."""
    
    if not os.path.exists(input_file):
        print(f"[LOG] File '{input_file}' not found, skipping conversion.")
        return

    # Load final_adjusted_log2_fpkm_uq.csv (with replaced headers)
    df = pd.read_csv(input_file)

    # Retain the first two columns as they are
    retained_columns = df.iloc[:, :2]  

    # Apply log2-to-linear transformation on all remaining columns (sample columns)
    transformed_data = df.iloc[:, 2:].apply(lambda x: (2 ** x) - 0.01)  

    # Concatenate retained columns with transformed data (ensuring column headers are intact)
    final_df = pd.concat([retained_columns, transformed_data], axis=1)

    # Save the transformed data with the same column headers
    final_df.to_csv(output_file, index=False)

    print(f"[LOG] Saved linearized data to '{output_file}'")
    # convert linear to gct file
def run_gct_conversion(input_file="step18_log2_to_linear.csv", output_file="expression.gct"):
    print(f"[LOG] Running GCT conversion on {input_file}...")

    # Load the input file
    df = pd.read_csv(input_file, sep=",", header=0)

    # Drop the "gene_id" column as it's not needed
    df = df.drop(columns=["gene_id"])

    # Step 1: Initialize output data with two blank rows
    output_data = []

    # Row 1: Add "#1,2" in the first cell, rest empty
    row_1 = [""] * (len(df.columns))
    row_1[0] = "#1,2"
    output_data.append(row_1)

    # Row 2: Add counts in column 1 (gene count) and column 2 (sample count)
    gene_symbol_count = len(df)
    sample_id_count = len(df.columns) - 1
    row_2 = [""] * (len(df.columns))
    row_2[0] = gene_symbol_count
    row_2[1] = sample_id_count
    output_data.append(row_2)

    # Step 2: Insert a blank column between column 1 (Symbol) and sample ID columns
    df.insert(1, "", "na")

    # Step 3: Add sample IDs in row 3
    sample_row = ["Symbol", "na"] + df.columns[2:].tolist()
    output_data.append(sample_row)

    # Step 4: Append the original matrix (gene symbols and expression values)
    for i in range(len(df)):
        row = df.iloc[i].tolist()
        output_data.append(row)

    # Convert to DataFrame for easier tab-delimited output
    final_df = pd.DataFrame(output_data)

    # Save to output file
    final_df.to_csv(output_file, sep="\t", index=False, header=False)

    print(f"[LOG] GCT conversion completed. Output saved as {output_file}")
    

# Main function to run all steps
def main():
    combined_data_file = "combined_data.csv"
    gene_reference_file = "gene_reference.csv"
    pickle_file = "combined_data.pkl"

    combined_data = load_combined_data(combined_data_file)
    gene_reference = load_gene_reference(gene_reference_file)

    filtered_data = filter_protein_coding_genes(combined_data, gene_reference)
    percentiles, muq = calculate_75th_percentile(filtered_data)

    # Apply gene size transformation and calculate median gene sizes after the transformation
    transformed_data, overall_mgs = apply_gene_size_transformation(filtered_data)

    # Normalize FPKM-UQ
    normalized_data = normalize_fpkm_uq(transformed_data, percentiles, muq, overall_mgs)

    # Log2 transformation of FPKM-UQ
    log2_transformed_data = log2_transform_fpkm_uq(normalized_data)

    # Calculate MAD and identify outliers
    mad_data, average_mad_values = calculate_mad_and_identify_outliers(
        log2_transformed_data
    )

    # Apply the FDR method to identify outliers
    average_mad_df = apply_fdr_method(average_mad_values)

    # Save the average MAD values with outlier status
    save_average_mad_with_outliers(average_mad_df)

    # Rescale cohorts, adjust log2 values, and save the adjusted data
    rescaled_data, delta = rescale_cohorts(
        log2_transformed_data,
        average_mad_df[average_mad_df["Outlier"] == "Yes"]["Sample"].tolist(),
    )
    adjusted_data = adjust_all_samples(log2_transformed_data, delta)

    # Replace negative log2 values with zero
    data_with_zeros = replace_negative_values_with_zero(adjusted_data)

    # Validate rescaling and document the global medians
    final_rescaled_data, new_global_median = validate_rescaling(
        data_with_zeros,
        average_mad_df[average_mad_df["Outlier"] == "Yes"]["Sample"].tolist(),
        overall_mgs,
    )

    # Save the final output with adjusted Log2 FPKM-UQ values
    save_final_output(final_rescaled_data)



    # 🔹 Step 3: Rename columns in final output files using Sample ID mapping
    # 🔹 Step 3: Rename columns in final output files using Sample ID mapping
    sample_id_mapping = load_sample_id_mapping(combined_data_file)

    if sample_id_mapping:
        rename_sample_rows("step8_outliers.csv", sample_id_mapping)  # ✅ Fixes row replacement
        rename_sample_columns("final_adjusted_log2_fpkm_uq.csv", sample_id_mapping)  # ✅ Keeps _FPKM_UQ
    else:
        print("[LOG] No Sample ID mapping found. Skipping renaming process.")


    # Clean up the Pickle file at the end to save space
    if os.path.exists(pickle_file):
        os.remove(pickle_file)
        print(f"[LOG] Deleted Pickle file: {pickle_file}")

    # Convert log2 values back to linear space
    convert_log2_to_linear("final_adjusted_log2_fpkm_uq.csv", "step18_log2_to_linear.csv")
# 🔹 Run GCT conversion as the final step
    run_gct_conversion("step18_log2_to_linear.csv", "expression.gct")





if __name__ == "__main__":
    main()