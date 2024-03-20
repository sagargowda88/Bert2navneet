 import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokens_labels):
        """
        Initialize a TextDataset.

        Args:
            tokens_labels (list): List of tuples containing tokenized text data and corresponding labels.

        This class is designed to work as a PyTorch Dataset, which means it can be used with PyTorch's DataLoader for efficient data loading during training and evaluation.
        """
        self.tokens_labels = tokens_labels  # List of tuples containing (tokenized text data, labels)

    def __len__(self):
        """
        Get the total number of data points in the dataset.

        Returns:
            int: Number of data points in the dataset.
        """
        return len(self.tokens_labels)

    def __getitem__(self, idx):
        """
        Get a specific data point (a pair of tokenized text data and its label) from the dataset.

        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            tuple: A tuple containing the tokenized text data and its label for the specified data point.
        """
        return self.tokens_labels[idx]
