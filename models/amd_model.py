from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn

class AMD_Model(nn.Module):
    def __init__(self, model_name, num_labels, lora_rank=8, lora_alpha=32, lora_dropout=0.1):
        super(AMD_Model, self).__init__()
        """
        Initializes the Adaptive Multi-Dialect (AMD) model using PEFT (LoRA).
        """
        # Load the base model (e.g., AraBERT)
        self.base_model = AutoModel.from_pretrained(model_name)

        # Define LoRA config for adapters
        self.peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS, 
            inference_mode=False, 
            r=lora_rank, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout
        )

        # Wrap the base model with LoRA adapters
        self.model = get_peft_model(self.base_model, self.peft_config)
        
        # Add a classification head
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        
        print("Initialized Adaptive Multi-Dialect Model (AMD)")
        self.model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        return logits
