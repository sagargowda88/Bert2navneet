 import torch
import config
from Source.data import TextDataset
from Source.utils import load_file, save_file
from sklearn.model_selection import train_test_split
from Source.model import BertClassifier, train, test

def main():
    # Load token, label, and label encoder files
    print("Loading the files...")
    data = load_file(config.data_path)
    labels = data[config.label_col].values.tolist()
    num_classes = len(set(labels))

    # Split data into train, valid, and test sets
    print("Splitting data into train, valid, and test sets...")
    train_data, test_data = train_test_split(data, test_size=0.2)
    train_data, valid_data = train_test_split(train_data, test_size=0.25)

    # Create PyTorch datasets
    print("Creating PyTorch datasets...")
    train_dataset = TextDataset(train_data)
    valid_dataset = TextDataset(valid_data)
    test_dataset = TextDataset(test_data)

    # Create data loaders
    print("Creating data loaders...")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create a model object
    print("Creating model object...")
    model = BertClassifier(config.dropout, num_classes)
    model_path = config.model_path

    # Move the model to the GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Define loss function (CrossEntropyLoss) and optimizer (Adam)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Train the model
    print("Training the model...")
    train(train_loader, valid_loader, model, criterion, optimizer, device, config.num_epochs, model_path)

    # Test the model
    print("Testing the model...")
    test(test_loader, model, criterion, device)

if __name__ == "__main__":
    main()
