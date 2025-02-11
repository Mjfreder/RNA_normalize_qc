import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to process a single file
def process_file(file_path, gene_symbol_df, file_id_to_sample_id):
    try:
        filename = os.path.basename(file_path)
        file_id_prefix = filename.split('.')[0]
        
        if file_id_prefix not in file_id_to_sample_id:
            print(f"Skipping file: {file_path}, ID not in mapping.")
            return None
        
        # Read the file
        file_df = pd.read_csv(file_path, delimiter='\t')
        print(f"Processing file: {file_path}")
        print(f"Columns in file: {list(file_df.columns)}")
        
        # Find the column containing 'gene_id'
        gene_id_column = next((col for col in file_df.columns if "gene_id" in col), None)
        if not gene_id_column:
            print(f"No column containing 'gene_id' found in {file_path}.")
            return None
        
        # Extract 'gene_id' from the identified column
        file_df = file_df.dropna(subset=['gene_id'])  # Remove NaN values
        file_df['gene_id'] = file_df['gene_id'].astype(str).str.split(' ').str[-1].str.split('.').str[0]
        print(f"Extracted 'gene_id' from column '{gene_id_column}'.")
        
        # Select relevant columns
        if not {'effective_length', 'expected_count'}.issubset(file_df.columns):
            print(f"Required columns missing in {file_path}.")
            return None
        
        selected_columns = file_df[['gene_id', 'effective_length', 'expected_count']]
        
        # Merge with gene_symbol_df
        merged_df = pd.merge(
            gene_symbol_df[['gene_id']],
            selected_columns,
            on='gene_id',
            how='inner'
        )
        print(f"Rows before merge: {len(selected_columns)}, after merge: {len(merged_df)}")
        
        sample_id = file_id_to_sample_id[file_id_prefix]
        merged_df.insert(1, 'Sample_ID', sample_id)
        return merged_df
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Main function
def main():
    current_directory = os.getcwd()
    inputs_directory = os.path.join(current_directory, 'Count_inputs')
    gene_symbol_path = os.path.join(current_directory, 'gene_reference.csv')
    mapping_file_path = os.path.join(current_directory, 'File_ID_Sample_ID_map.csv')
    
    gene_symbol_df = pd.read_csv(gene_symbol_path)
    gene_symbol_df['gene_id'] = gene_symbol_df['gene_id'].astype(str).str.split('.').str[0]
    mapping_df = pd.read_csv(mapping_file_path)
    file_id_to_sample_id = dict(zip(mapping_df['File Id'], mapping_df['Sample ID']))
    
    if not os.path.exists(inputs_directory):
        print(f"Count_inputs directory '{inputs_directory}' does not exist.")
        return
    
    # Find all files in the inputs directory and match them to valid File IDs
    input_files = [
        os.path.join(inputs_directory, f) for f in os.listdir(inputs_directory)
        if f.split('.')[0] in file_id_to_sample_id  # Only select files with matching prefixes
    ]
    
    if not input_files:
        print("No matching files found in the Count_inputs directory.")
        return
    
    print(f"Found {len(input_files)} matching files in the inputs directory.")
    dataframes = []
    first_gene_id_column = None  # Placeholder to store the gene_id column from the first file
    
    with ThreadPoolExecutor() as executor:  # Switched to ThreadPoolExecutor
        futures = {
            executor.submit(process_file, file_path, gene_symbol_df, file_id_to_sample_id): file_path
            for file_path in input_files
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    if first_gene_id_column is None:
                        first_gene_id_column = result['gene_id'].reset_index(drop=True)
                    dataframes.append(result.drop(columns=['gene_id']))
            except Exception as e:
                print(f"Error in future: {e}")
    
    print(f"Number of valid DataFrames: {len(dataframes)}")
    if dataframes:
        try:
            # Combine the DataFrames horizontally (concatenating columns)
            combined_df = pd.concat(dataframes, axis=1)
            
            # Insert the gene_id column as the first column
            combined_df.insert(0, 'gene_id', first_gene_id_column)
            
            print("combined_data created successfully.")
            print("combined_data preview:")
            print(combined_df.head())
            
            output_path = os.path.join(current_directory, 'combined_data.csv')
            combined_df.to_csv(output_path, index=False)
            print(f"combined_data saved to '{output_path}'")
        except Exception as e:
            print(f"Error during concatenation or saving: {e}")
    else:
        print("No valid DataFrames were processed.")

if __name__ == "__main__":
    main()
