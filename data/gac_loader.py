
from utils.data_loader import BaseArabicDataset

class GACDataset(BaseArabicDataset):
    def _load_data(self, file_path):
        # Simplified loader for Gulf Arabic Corpus
        sentences = ["A sentence from the Gulf Arabic Corpus.", "Another sentence here."]
        labels = [[0, 1, 2, 3, 4, 5], [6, 7, 8]] # Placeholder
        return sentences, labels
