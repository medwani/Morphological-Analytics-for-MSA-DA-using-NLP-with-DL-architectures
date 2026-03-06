
from utils.data_loader import BaseArabicDataset

class MADDataset(BaseArabicDataset):
    def _load_data(self, file_path):
        # Simplified loader for Maghrebi Arabic Dataset
        sentences = ["A sentence from the Maghrebi Arabic Dataset.", "Another one."]
        labels = [[0, 1, 2, 3, 4, 5], [6, 7]] # Placeholder
        return sentences, labels
