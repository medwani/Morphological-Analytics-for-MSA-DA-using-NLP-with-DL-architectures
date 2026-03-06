
from utils.data_loader import BaseArabicDataset

class GigawordDataset(BaseArabicDataset):
    def _load_data(self, file_path):
        # Simplified loader for Gigaword
        sentences = ["Sentence from Gigaword.", "Another one."]
        labels = [[0, 1, 2], [3, 4]] # Placeholder
        return sentences, labels
