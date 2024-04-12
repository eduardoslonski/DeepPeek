from transformers import AutoTokenizer, AutoModelForCausalLM

import torch.nn.functional as F
import numpy as np

class Model:
    def __init__(self):
        self.load_model()

    def load_model(self):
        model_hf = "EleutherAI/pythia-1B-deduped"

        self.tokenizer = AutoTokenizer.from_pretrained(model_hf)
        self.model = AutoModelForCausalLM.from_pretrained(model_hf)
        self.device = "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

        self.model_embed = self.model.gpt_neox.embed_in
        self.max_context_length = self.model.config.max_position_embeddings
        self.model_layers = self.model.gpt_neox.layers
        self.d_model = self.model.config.hidden_size
        self.n_attn_heads = self.model.config.num_attention_heads
        self.d_attn_head = self.d_model // self.n_attn_heads
        self.rotary_pct = self.model.config.rotary_pct
        self.n_layers = self.model.config.num_hidden_layers
        self.vocab_size = self.model.config.vocab_size
        self.model_name = self.model.config.name_or_path

    def model_config(self):
        return {"model_name": self.model_name, "d_model": self.d_model ,"n_layers": self.n_layers,
                 "n_attn_heads": self.n_attn_heads, "d_attn_head": self.d_attn_head}
    
    def tokenizer_decode(self, idx):
        return self.tokenizer.decode(idx)