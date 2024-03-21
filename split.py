import pandas as pd

# Load the CSV file
df = pd.read_csv("your_file.csv")

# Concatenate all column values for each row
concatenated_text = df.apply(lambda row: ' '.join(row.astype(str)), axis=1)

# Add the concatenated text as a new column in the DataFrame
df['concatenated_text'] = concatenated_text

# Print the concatenated text for the first row
print(df['concatenated_text'][0])

# Save the modified DataFrame to a CSV file named "bmg.csv"
df.to_csv("bmg.csv", index=False)
