import re
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
from Source.utils import load_file
from Source.model import BertClassifier
import config

def preprocess_test_set(test_data):
    # Tokenize each text column separately
    all_tokens = []
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    for col in config.text_cols:
        input_text = test_data[col].apply(lambda x: str(x))
        input_text = input_text.apply(lambda x: x.lower())
        input_text = input_text.apply(lambda x: re.sub(r"[^\w\d'\s]+", " ", x))
        input_text = input_text.apply(lambda x: re.sub("\d+", "", x))
        input_text = input_text.apply(lambda x: re.sub(r'[x]{2,}', "", x))
        input_text = input_text.apply(lambda x: re.sub(' +', ' ', x))
        tokens = [tokenizer(i, padding="max_length",
                            max_length=config.seq_len, truncation=True,
                            return_tensors="pt")
                  for i in tqdm(input_text)]
        all_tokens.extend(tokens)
    
    # Combine the tokens
    combined_input_ids = torch.cat([tokens["input_ids"] for tokens in all_tokens], dim=1)
    combined_attention_mask = torch.cat([tokens["attention_mask"] for tokens in all_tokens], dim=1)

    return combined_input_ids, combined_attention_mask

def main():
    # Read the test data file
    print("Loading test data...")
    test_data = pd.read_csv(config.test_data_path)

    # Preprocess the test set
    print("Preprocessing test set...")
    input_ids, attention_mask = preprocess_test_set(test_data)

    # Load the label encoder to map predicted indices back to class labels
    label_encoder = load_file(config.label_encoder_path)
    num_classes = len(label_encoder.classes_)

    # Load the pre-trained BERT-based classifier model
    print("Loading the model...")
    model = BertClassifier(config.dropout, num_classes)
    model.load_state_dict(torch.load(config.model_path, map_location=torch.device('cpu')))

    # Perform a forward pass to make predictions
    print("Making predictions...")
    out = model(input_ids, attention_mask)

    # Find the predicted classes
    predictions = [label_encoder.classes_[torch.argmax(pred).item()] for pred in out]

    # Add predictions to the test data
    test_data['Predicted_Class'] = predictions

    # Save the test data with predictions
    print("Saving test data with predictions...")
    test_data.to_csv(config.predict_output_path, index=False)

if __name__ == "__main__":
    main()
