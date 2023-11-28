import pandas as pd

# Assuming your tab-delimited .txt file is named 'input.txt'
input_txt_file = './datasetFinalStable-Beluga2.txt'

# Read the tab-delimited .txt file into a pandas DataFrame
df = pd.read_csv(input_txt_file, delimiter='\t', header=None, names=['text'])

# Assuming your .parquet file will be named 'output.parquet'
output_parquet_file = './datasetFinalStable-Beluga2.parquet'

# Write the DataFrame to a .parquet file with the specified header
df.to_parquet(output_parquet_file, index=False)

print(f'{input_txt_file} has been successfully converted to {output_parquet_file} with "text" as the header.')
