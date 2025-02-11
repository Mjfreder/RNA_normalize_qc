import pandas as pd
import numpy as np
import os

# Load input files
fpkm_df = pd.read_csv("final_adjusted_log2_fpkm_uq.csv")
gc_df = pd.read_csv("Gene_GC.csv")
reference_df = pd.read_csv("Reference_ave.csv")

# Drop 'gene_id' column from fpkm_df
if 'gene_id' in fpkm_df.columns:
    fpkm_df = fpkm_df.drop(columns=['gene_id'])

# Merge dataframes on 'Symbol', keeping only genes present in all three files
merged_df = fpkm_df.merge(gc_df, on="Symbol", how="inner").merge(reference_df, on="Symbol", how="inner")

# Track dropped gene symbols
all_symbols = set(fpkm_df["Symbol"]) | set(gc_df["Symbol"]) | set(reference_df["Symbol"])
merged_symbols = set(merged_df["Symbol"])
dropped_symbols = all_symbols - merged_symbols

# Ensure numeric values in '% GC', 'Reference', and sample columns
invalid_symbols = set()
for col in merged_df.columns[1:]:  # Skip 'Symbol'
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
    invalid_symbols.update(merged_df.loc[merged_df[col].isna(), 'Symbol'])

# Combine all dropped symbols
final_dropped_symbols = dropped_symbols | invalid_symbols

# Save dropped gene symbols before removing them
if final_dropped_symbols:
    dropped_genes_path = os.path.abspath("dropped_genes.txt")
    with open(dropped_genes_path, "w") as f:
        f.write("\n".join(sorted(final_dropped_symbols)))
    print(f"Dropped gene symbols saved to {dropped_genes_path}")

# Drop rows with invalid numeric values
merged_df = merged_df.dropna()

# Save final Plot.csv with correct column order
column_order = ["Symbol", "% GC", "Reference"] + [col for col in fpkm_df.columns if col != "Symbol"]
merged_df = merged_df[column_order]

# Sort by '% GC' column in ascending order
merged_df = merged_df.sort_values(by="% GC")

# Save the sorted file
plot_csv_path = os.path.abspath("Plot.csv")
merged_df.to_csv(plot_csv_path, index=False)

print(f"Plot.csv generated and sorted successfully! Saved at {plot_csv_path}")
