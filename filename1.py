import os

# Define the network folder path and output file name
folder_path = os.path.join(os.getcwd(), "Count_inputs")  # Look inside "Counts_input" in the script directory
output_file = "file_names.txt"

# Get the list of all files in the folder
try:
    file_names = os.listdir(folder_path)

    # Write the file names to the output text file
    with open(output_file, "w") as f:
        for file_name in file_names:
            truncated_name = file_name.split('.')[0]  # Keep only the prefix before the first dot
            f.write(truncated_name + "\n")

    print(f"File names successfully saved to {output_file}")
except FileNotFoundError:
    print(f"The folder path {folder_path} does not exist. Please check the path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")