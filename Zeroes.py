from scipy.stats import binom
import numpy as np
import pandas as pd

# Load the file to examine its structure
file_path = 'Zeros.csv'
data = pd.read_csv(file_path)

# Display the first few rows to understand its format
data.head()

# Define the function for Benjamini-Hochberg correction
def benjamini_hochberg(p_values, fdr_threshold=0.1):
    # Sort p-values and get their corresponding indices
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.array(p_values)[sorted_indices]

    # Number of hypotheses
    m = len(p_values)

    # Calculate the Benjamini-Hochberg thresholds
    bh_thresholds = np.array([(i + 1) / m * fdr_threshold for i in range(m)])

    # Initialize the adjusted p-values
    adjusted_p_values = np.zeros(m)

    # Calculate the adjusted p-values
    for i in range(m):
        adjusted_p_values[i] = sorted_p_values[i] * m / (i + 1)

    # Ensure adjusted p-values are non-decreasing
    adjusted_p_values = np.minimum.accumulate(adjusted_p_values[::-1])[::-1]

    # Ensure p-values don't exceed 1
    adjusted_p_values = np.minimum(adjusted_p_values, 1.0)

    # Return the p-values in their original order
    return adjusted_p_values[np.argsort(sorted_indices)]

# Calculate the binomial probability for each gene and get p-values
p_values = []

for index, row in data.iterrows():
    # Calculate the binomial probability of getting >= Observed_zeros successes
    observed_successes = row['Observed_zeros']
    total_samples = row['Total_samples']
    probability = row['Probability']
    
    # Calculate cumulative probability of getting at least observed_successes
    p_value = binom.sf(observed_successes - 1, total_samples, probability)
    p_values.append(p_value)

# Apply Benjamini-Hochberg correction
adjusted_p_values = benjamini_hochberg(p_values, fdr_threshold=0.1)

# Add raw and adjusted p-values to the dataframe
data['Raw_P_value'] = p_values
data['Adjusted_P_value'] = adjusted_p_values

# Save the updated dataframe to a CSV file
output_path = 'Zeros_with_P_values.csv'
data.to_csv(output_path, index=False)

# Display the updated dataframe
print(data)
