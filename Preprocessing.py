 import re
import config
import pandas as pd
from tqdm import tqdm
from Source.utils import save_file
from transformers import BertTokenizer

def main():
    # Read the data file
    print("Processing data file...")
    data = pd.read_csv(config.data_path)
    
    # Drop rows where any of the text columns are empty
    data.dropna(subset=config.text_cols + [config.label_col], inplace=True)
    
    # Process the text columns
    input_text = data[config.text_cols].apply(lambda x: ' '.join(x), axis=1)
    
    # Convert text to lowercase
    print("Converting text to lowercase...")
    input_text = [i.lower() for i in tqdm(input_text)]
    
    # Remove punctuations except apostrophe
    print("Removing punctuations in text...")
    input_text = [re.sub(r"[^\w\d'\s]+", " ", i) for i in tqdm(input_text)]
    
    # Remove digits
    print("Removing digits in text...")
    input_text = [re.sub("\d+", "", i) for i in tqdm(input_text)]
    
    # Remove more than one consecutive instance of 'x'
    print("Removing 'xxxx...' in text")
    input_text = [re.sub(r'[x]{2,}', "", i) for i in tqdm(input_text)]
    
    # Replace multiple spaces with a single space
    print("Removing additional spaces in text...")
    input_text = [re.sub(' +', ' ', i) for i in tqdm(input_text)]
    
    # Tokenize the text using a BERT tokenizer
    print("Tokenizing the text...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    all_tokens = []
    for text in tqdm(input_text):
        tokens = tokenizer.encode_plus(text, padding="max_length", max_length=config.seq_len,
                                        truncation=True, return_tensors="pt")
        all_tokens.append(tokens["input_ids"].squeeze())  # Use only input_ids
    
    # Save the tokens for later use
    save_file(config.tokens_path, all_tokens)

if __name__ == "__main__":
    main()
