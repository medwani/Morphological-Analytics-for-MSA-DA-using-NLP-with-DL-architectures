from transformers import AutoModelForTokenClassification

def create_transformer_model(model_name, num_labels):
    """
    Loads a pre-trained transformer model from Hugging Face and adapts it for token classification.
    """
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    return model
