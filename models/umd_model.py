from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

class UMD_Model:
    def __init__(self, model_name, num_labels, dialects):
        """
        Initializes a single transformer model to handle all dialects.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Add dialect tokens to the tokenizer
        self.dialect_tokens = [f"[{d}]" for d in dialects]
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.dialect_tokens})
        
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.resize_token_embeddings(len(self.tokenizer))
        print("Initialized Unified Multi-Dialect Model (UMD)")

    def predict(self, text, dialect):
        if f"[{dialect}]" not in self.dialect_tokens:
            raise ValueError(f"Unknown dialect: {dialect}")

        # Prepend the dialect token to the input text
        text_with_dialect = f"[{dialect}] {text}"
        
        inputs = self.tokenizer(text_with_dialect, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        predictions = torch.argmax(logits, dim=2)
        return {"predictions": predictions}
