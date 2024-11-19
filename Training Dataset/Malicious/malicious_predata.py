import csv
import glob

# Input and output file paths
input_pattern = "*.csv"  # Pattern to match all input files
output_file = "malicious_dataset.csv"

# Find files matching the pattern
input_files = glob.glob(input_pattern)

# Check if files were found
if not input_files:
    raise FileNotFoundError(f"No files found matching the pattern: {input_pattern}")

data = []

# Process each file
for input_file in input_files:
    with open(input_file, "r", encoding="utf-8") as file:
        reader = file.readlines()
        for i, line in enumerate(reader):
            sentence = line.strip()  # Remove extra whitespace
            data.append([len(data), sentence, 1])  # Global sequence number and label 1

# Write the transformed data to the output file
with open(output_file, "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["", "Sentence", "Label"])  # Header row
    writer.writerows(data)

print(f"Transformation complete. {len(data)} rows saved to {output_file}")
