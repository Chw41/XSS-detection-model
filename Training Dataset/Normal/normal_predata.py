import csv
import glob

# Directory containing all CSV files
input_files = glob.glob('DeepXSS_nomal.csv')  # Matches all .csv files in the Malicious folder
output_file = 'normal_dataset.csv'  # The combined output file

# Function to determine if a sentence contains XSS indicators
def is_xss(sentence):
    xss_keywords = ['<script>', 'alert(', 'onmouseover=', '<ScRiPt>', 'document.cookie']
    return any(keyword in sentence.lower() for keyword in xss_keywords)

# Combine and process all CSV files
with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    csv_writer = csv.writer(outfile)
    
    # Write the header row
    csv_writer.writerow(['', 'Sentence', 'Label'])
    
    row_index = 0
    for file in input_files:
        with open(file, 'r', encoding='utf-8') as infile:
            csv_reader = csv.reader(infile)
            for row in csv_reader:
                # Skip empty rows
                if not row or len(row) == 0:
                    continue
                
                # Safely process rows with content
                sentence = row[0]
                label = 1 if is_xss(sentence) else 0
                csv_writer.writerow([row_index, sentence, label])
                row_index += 1

print(f"Conversion complete! All results are saved to {output_file}")
