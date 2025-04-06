import pandas as pd

# Load the three CSV files
malicious_df = pd.read_csv('Malicious/malicious_dataset.csv')
normal_df = pd.read_csv('Normal/normal_dataset.csv')
xss_df = pd.read_csv('XSS_dataset.csv')

# Remove any unnecessary 'Unnamed' columns
malicious_df = malicious_df.loc[:, ~malicious_df.columns.str.contains('^Unnamed')]
normal_df = normal_df.loc[:, ~normal_df.columns.str.contains('^Unnamed')]
xss_df = xss_df.loc[:, ~xss_df.columns.str.contains('^Unnamed')]

# Combine all datasets
combined_df = pd.concat([malicious_df, normal_df, xss_df], ignore_index=True)

# Reset index to ensure proper formatting
combined_df.reset_index(inplace=True)
combined_df.rename(columns={'index': ''}, inplace=True)

# Save the final file
combined_df.to_csv('final_dataset.csv', index=False, encoding='utf-8')

print("Merge complete! The file has been saved as final_dataset.csv")
