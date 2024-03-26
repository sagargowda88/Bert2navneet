import pandas as pd

# Load the CSV file
df = pd.read_csv("your_file.csv")

# Concatenate all column values except the last column for each row
concatenated_text = df.iloc[:, :-1].apply(lambda row: ' '.join(row.astype(str)), axis=1)

# Create a new DataFrame with the concatenated text and the last column
new_df = pd.concat([concatenated_text, df.iloc[:, -1]], axis=1)

# Print the first row of the new DataFrame
print(new_df.head(1))

# Save the new DataFrame to a CSV file named "new_bmg.csv"
new_df.to_csv("new_bmg.csv", index=False)
