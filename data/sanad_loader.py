
from utils.data_loader import BaseArabicDataset

class SANADDataset(BaseArabicDataset):
    def _load_data(self, file_path):
        # Simplified loader for SANAD
        sentences = ["A sentence from the SANAD corpus.", "And another."]
        labels = [[0, 1, 2, 3, 4], [5, 6]] # Placeholder
        return sentences, labels
