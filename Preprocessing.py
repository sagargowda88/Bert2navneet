# main.py

import re
import config
import pandas as pd
from tqdm import tqdm
from Source.utils import save_file
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

def main():
    # Read the data file
    print("Processing data file...")
    data = pd.read_csv(config.data_path)
    
    # Drop rows where any of the text columns are empty
    data.dropna(subset=config.text_col_names + [config.label_col], inplace=True)
    
    # Replace duplicate labels using the provided mapping
    data.replace({config.label_col: config.product_map}, inplace=True)
    
    # Encode the label column using sklearn's LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(data[config.label_col])
    labels = label_encoder.transform(data[config.label_col])
    
    # Save the encoded labels and label encoder for later use
    save_file(config.labels_path, labels)
    save_file(config.label_encoder_path, label_encoder)

    # Process text columns
    all_tokens = []
    for text_col_name in config.text_col_names:
        input_text = data[text_col_name]
        
        # Convert text to lowercase
        print(f"Converting {text_col_name} to lowercase...")
        input_text = [i.lower() for i in tqdm(input_text)]
        
        # Remove punctuations except apostrophe
        print(f"Removing punctuations in {text_col_name}...")
        input_text = [re.sub(r"[^\w\d'\s]+", " ", i) for i in tqdm(input_text)]
        
        # Remove digits
        print(f"Removing digits in {text_col_name}...")
        input_text = [re.sub("\d+", "", i) for i in tqdm(input_text)]
        
        # Remove more than one consecutive instance of 'x'
        print(f"Removing 'xxxx...' in {text_col_name}")
        input_text = [re.sub(r'[x]{2,}', "", i) for i in tqdm(input_text)]
        
        # Replace multiple spaces with a single space
        print(f"Removing additional spaces in {text_col_name}...")
        input_text = [re.sub(' +', ' ', i) for i in tqdm(input_text)]
        
        # Tokenize the text using a BERT tokenizer
        print(f"Tokenizing {text_col_name}...")
        input_text = input_text[:1000]  # Limit the number of texts for faster processing
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        tokens = [tokenizer(i, padding="max_length", max_length=config.seq_len,
                            truncation=True, return_tensors="pt")
                  for i in tqdm(input_text)]
        
        all_tokens.extend(tokens)

    # Save all tokens to a single file
    save_file(config.tokens_path, np.array(all_tokens))

if __name__ == "__main__":
    main()
