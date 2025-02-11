import pandas as pd
import numpy as np

# Input and output file names
input_file = "step18_log2_to_linear.csv"
output_file = "expression.gct"

# Load the input file
# The file is expected to be a CSV with the first two columns: "gene_id" and "Symbol"
df = pd.read_csv(input_file, sep=",", header=0)

# Drop the "gene_id" column as it's not needed
df = df.drop(columns=["gene_id"])

# Step 1: Initialize output data with two blank rows
output_data = []

# Row 1: Add "#1,2" in the first cell, rest empty
row_1 = [""] * (len(df.columns))  # No extra column since we already dropped "gene_id"
row_1[0] = "#1,2"
output_data.append(row_1)

# Row 2: Add counts in column 1 (gene count) and column 2 (sample count)
gene_symbol_count = len(df)  # Number of genes
sample_id_count = len(df.columns) - 1  # Exclude "Symbol" column
row_2 = [""] * (len(df.columns))
row_2[0] = gene_symbol_count
row_2[1] = sample_id_count
output_data.append(row_2)

# Step 2: Insert a blank column between column 1 (Symbol) and sample ID columns
df.insert(1, "", "na")  # Add 'na' in the blank column starting from row 2

# Step 3: Add sample IDs in row 3
sample_row = ["Symbol", "na"] + df.columns[2:].tolist()  # Columns after 'Symbol' and blank
output_data.append(sample_row)

# Step 4: Append the original matrix (gene symbols and expression values)
for i in range(len(df)):
    row = df.iloc[i].tolist()
    output_data.append(row)

# Convert to DataFrame for easier tab-delimited output
final_df = pd.DataFrame(output_data)

# Save to output file
final_df.to_csv(output_file, sep="\t", index=False, header=False)

print(f"File successfully saved as {output_file}")
