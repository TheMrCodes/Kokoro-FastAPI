# https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py
from typing import Optional
import torch
from transformers import AlbertConfig, AlbertModel


class CustomAlbert(AlbertModel):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ):
        # Call the original forward method
        outputs = super().forward(
            input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # Only return the last_hidden_state
        return outputs.last_hidden_state

def load_plbert():
    plbert_config = {'vocab_size': 178, 'hidden_size': 768, 'num_attention_heads': 12, 'intermediate_size': 2048, 'max_position_embeddings': 512, 'num_hidden_layers': 12, 'dropout': 0.1}
    albert_base_configuration = AlbertConfig(**plbert_config)
    bert = CustomAlbert(albert_base_configuration)
    return bert
