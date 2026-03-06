
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import os

class BaseArabicDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sentences, self.labels = self._load_data(file_path)

    def _load_data(self, file_path):
        """Placeholder method to be overridden by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding=\'max_length\',
            truncation=True,
            return_attention_mask=True,
            return_tensors=\'pt\',
        )

        # This is a simplified example. In a real scenario, you would have a more complex label encoding scheme.
        # For simplicity, we\'ll just return the raw labels here.
        return {
            \'input_ids\': encoding[\'input_ids\'].flatten(),
            \'attention_mask\': encoding[\'attention_mask\'].flatten(),
            \'labels\': torch.tensor(labels, dtype=torch.long) # Placeholder
        }

def create_data_loader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4
    )
