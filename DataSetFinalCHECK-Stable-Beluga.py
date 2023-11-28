import os

input_path = './dataset2.txt'
output_path = './datasetFinalStable-Beluga.txt'
temp_path = './temp_dataset.txt'

try:
    with open(input_path, 'r', encoding='utf-8') as file:
        input_lines = file.readlines()
except FileNotFoundError as e:
    print(f"Error reading the input file: {e}")
    exit()

transformed_lines = []
errors = []

for line_number, line in enumerate(input_lines, start=1):
    if all(line.count(substring) == 1 for substring in ["### Comment:", "### REPLY:", "### END."]):
        transformed_lines.append(line.strip())
    else:
        errors.append(line_number)
        print(f"Error: Missing or multiple substrings in line {line_number} of the original file: {line}")

transformed_text = '\n'.join(filter(None, transformed_lines))

try:
    with open(temp_path, 'w', encoding='utf-8') as file:
        file.write(transformed_text)
except IOError as e:
    print(f"Error writing to the temporary file: {e}")
    exit()

try:
    with open(temp_path, 'r', encoding='utf-8') as temp_file:
        temp_lines = temp_file.readlines()
except FileNotFoundError as e:
    print(f"Error reading the temporary file: {e}")
    exit()

try:
    with open(output_path, 'w', encoding='utf-8') as final_file:
        final_file.writelines(temp_lines)
except IOError as e:
    print(f"Error writing to the final output file: {e}")
    exit()

print('Transformation complete. Result saved to datasetFinal.txt.')

# Delete the temporary file
try:
    os.remove(temp_path)
except OSError as e:
    print(f"Error deleting the temporary file: {e}")
    exit()

if errors:
    print(f"Deleting lines with errors from the original file.")

    # Remove lines with errors from the original file
    try:
        with open(input_path, 'w', encoding='utf-8') as original_file:
            original_file.writelines(line for i, line in enumerate(input_lines, start=1) if i not in errors)
    except IOError as e:
        print(f"Error updating the original file: {e}")
        exit()

    print('Lines with errors deleted from the original file.')