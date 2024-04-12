import torch
import torch.nn.functional as F
import numpy as np
from app.models.rope import RotaryEmbedding, apply_rotary_pos_emb
from app.utils.utils import get_bounds_normalization, get_bins_histogram, create_histogram, get_outliers, get_median_histogram_activations_layers

class Samples():
    def __init__(self, model_class):
        self.model_class = model_class
        self.device = "cpu"
        self.model = self.model_class.model.to(self.device)
        self.tokenizer = self.model_class.tokenizer
        self.n_attn_heads = self.model_class.n_attn_heads
        self.d_attn_head = self.model_class.d_attn_head
        self.d_model = self.model_class.d_model
        self.n_layers = self.model_class.n_layers
        self.rotary_pct = self.model_class.rotary_pct
        self.max_context_length = self.model_class.max_context_length
        self.model_name = self.model_class.model_name

        self.file_samples = "app/samples_data"
        
        self.reset()

    def add(self, sample):
        self.samples.append(sample)
        self.samples_tokenized.append(self.tokenizer.encode(sample, return_tensors="pt").to(self.model_class.device))

    def replace(self, sample, sample_idx):
        self.samples[sample_idx] = sample
        self.samples_tokenized[sample_idx] = self.tokenizer.encode(sample, return_tensors="pt").to(self.model_class.device)

    def reset(self):
        self.samples = []
        self.samples_tokenized = []
        self.samples_untokenized = []
        
        self.list_attentions = []
        self.logits = []

        self.hooks = []

        self.activations_bounds_histogram = self._create_histogram_dict()
        self.activations_bins_histogram = self._create_histogram_dict()
        self.activations_median_histogram = self._create_histogram_dict()

        self.similarities_token = self._create_histogram_dict()
        self.similarities_previous = self._create_histogram_dict()
        
    def _create_histogram_dict(self):
        rope_mode_list = ["full", "applied", "not_applied"]
        return {
            "embed": None,
            "input": [None] * self.n_layers,
            "input_layernorm": [None] * self.n_layers,
            "q": [[{rope_mode: None for rope_mode in rope_mode_list} for _ in range(self.n_attn_heads)] for _ in range(self.n_layers)],
            "k": [[{rope_mode: None for rope_mode in rope_mode_list} for _ in range(self.n_attn_heads)] for _ in range(self.n_layers)],
            "q_rope": [[{rope_mode: None for rope_mode in rope_mode_list} for _ in range(self.n_attn_heads)] for _ in range(self.n_layers)],
            "k_rope": [[{rope_mode: None for rope_mode in rope_mode_list} for _ in range(self.n_attn_heads)] for _ in range(self.n_layers)],
            "v": [[None for _ in range(self.n_attn_heads)] for _ in range(self.n_layers)],
            "o": [[None for _ in range(self.n_attn_heads)] for _ in range(self.n_layers)],
            "o_mm_dense": [[None for _ in range(self.n_attn_heads)] for _ in range(self.n_layers)],
            "dense_attention": [None] * self.n_layers,
            "dense_attention_residual": [None] * self.n_layers,
            "post_attention_layernorm": [None] * self.n_layers,
            "mlp_h_to_4": [None] * self.n_layers,
            "mlp_4_to_h": [None] * self.n_layers,
            "output": [None] * self.n_layers,
        }

    def _register_hooks(self, sample_idx):
        def register_forward_hook_embed(embed):
            def hook_fn(module, input, output):
                output.cpu().numpy().tofile(f"{self.file_samples}/sample_{sample_idx}/embed/embed.bin")
            hook = embed.register_forward_hook(hook_fn)
            self.hooks.append(hook)
        
        def register_forward_hook_input_layernorm(layer, layer_idx):
            def hook_fn(module, input, output):
                input[0].cpu().numpy().tofile(f"{self.file_samples}/sample_{sample_idx}/input/input_layer_{layer_idx}.bin")
                output.cpu().numpy().tofile(f"{self.file_samples}/sample_{sample_idx}/input_layernorm/input_layernorm_layer_{layer_idx}.bin")
            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)
        
        def register_forward_hook_qkv(layer, layer_idx):
            def hook_fn(module, input, output):
                idx_qkv = {"q": 0, "k": 1, "v": 2}
                q = output.view(-1, self.n_attn_heads, self.d_attn_head * 3)[:, :, idx_qkv["q"]*self.d_attn_head:idx_qkv["q"]*self.d_attn_head+self.d_attn_head]
                k = output.view(-1, self.n_attn_heads, self.d_attn_head * 3)[:, :, idx_qkv["k"]*self.d_attn_head:idx_qkv["k"]*self.d_attn_head+self.d_attn_head]
                v = output.view(-1, self.n_attn_heads, self.d_attn_head * 3)[:, :, idx_qkv["v"]*self.d_attn_head:idx_qkv["v"]*self.d_attn_head+self.d_attn_head]
                q.cpu().reshape(1, -1, self.d_model).numpy().tofile(f"{self.file_samples}/sample_{sample_idx}/q/q_layer_{layer_idx}.bin")
                k.cpu().reshape(1, -1, self.d_model).numpy().tofile(f"{self.file_samples}/sample_{sample_idx}/k/k_layer_{layer_idx}.bin")
                v.cpu().reshape(1, -1, self.d_model).numpy().tofile(f"{self.file_samples}/sample_{sample_idx}/v/v_layer_{layer_idx}.bin")

                #rope
                q_rope = q.clone().unsqueeze(1)
                k_rope = k.clone().unsqueeze(1)

                rotary_ndims = int(
                    self.d_attn_head * self.rotary_pct
                )

                rotary_emb = RotaryEmbedding(
                    dim=rotary_ndims,
                    max_seq_len=self.max_context_length,
                    precision=torch.float32,
                )

                query_rot, query_pass = (
                    q_rope[..., : rotary_ndims],
                    q_rope[..., rotary_ndims :],
                )
                key_rot, key_pass = (
                    k_rope[..., : rotary_ndims],
                    k_rope[..., rotary_ndims :],
                )

                seq_len = k_rope.shape[0]
                offset = 0

                cos, sin = rotary_emb(v, seq_len=seq_len)
                q_rope, k_rope = apply_rotary_pos_emb(
                    query_rot, key_rot, cos, sin, offset=offset
                )

                q_rope = torch.cat((q_rope, query_pass), dim=-1)
                k_rope = torch.cat((k_rope, key_pass), dim=-1)

                q_rope = q_rope.squeeze(1).view(1, seq_len, -1)
                k_rope = k_rope.squeeze(1).view(1, seq_len, -1)

                q_rope.cpu().numpy().tofile(f"{self.file_samples}/sample_{sample_idx}/q_rope/q_rope_layer_{layer_idx}.bin")
                k_rope.cpu().numpy().tofile(f"{self.file_samples}/sample_{sample_idx}/k_rope/k_rope_layer_{layer_idx}.bin")

            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)
        
        def register_forward_hook_dense_attention(layer, layer_idx):
            def hook_fn(module, input, output):
                input[0].cpu().numpy().tofile(f"{self.file_samples}/sample_{sample_idx}/o/o_layer_{layer_idx}.bin")
                o = input[0].detach().view(-1, self.n_attn_heads, self.d_attn_head)
                weights = self.model.gpt_neox.layers[layer_idx].attention.dense.weight.data.detach().T.view(self.n_attn_heads, -1, self.d_model)

                result = torch.empty(o.shape[0], self.n_attn_heads, self.d_model)

                for i in range(self.n_attn_heads):
                    result[:, i, :] = torch.matmul(o[:, i, :], weights[i])

                result.cpu().numpy().tofile(f"{self.file_samples}/sample_{sample_idx}/o_mm_dense/o_mm_dense_layer_{layer_idx}.bin")
                
                output.cpu().numpy().tofile(f"{self.file_samples}/sample_{sample_idx}/dense_attention/dense_attention_layer_{layer_idx}.bin")
            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)
        
        def register_forward_hook_input_post_attention_layernorm(layer, layer_idx):
            def hook_fn(module, input, output):
                input[0].cpu().numpy().tofile(f"{self.file_samples}/sample_{sample_idx}/dense_attention_residual/dense_attention_residual_layer_{layer_idx}.bin")
                output.cpu().numpy().tofile(f"{self.file_samples}/sample_{sample_idx}/post_attention_layernorm/post_attention_layernorm_layer_{layer_idx}.bin")
            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)
        
        def register_forward_hook_mlp_h_to_4(layer, layer_idx):
            def hook_fn(module, input, output):
                output.cpu().numpy().tofile(f"{self.file_samples}/sample_{sample_idx}/mlp_h_to_4/mlp_h_to_4_layer_{layer_idx}.bin")
            hook = layer.register_forward_hook(hook_fn)
            hook = self.hooks.append(hook)
        
        def register_forward_hook_mlp_4_to_h(layer, layer_idx):
            def hook_fn(module, input, output):
                output.cpu().numpy().tofile(f"{self.file_samples}/sample_{sample_idx}/mlp_4_to_h/mlp_4_to_h_layer_{layer_idx}.bin")
            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)
        
        def register_forward_hook_output(layer, layer_idx):
            def hook_fn(module, input, output):
                output, _, _ = output
                output.cpu().numpy().tofile(f"{self.file_samples}/sample_{sample_idx}/output/output_layer_{layer_idx}.bin")
            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)

        register_forward_hook_embed(self.model_class.model_embed)
            
        for layer_idx, layer in enumerate(self.model_class.model_layers):
            register_forward_hook_input_layernorm(layer.input_layernorm, layer_idx)
            register_forward_hook_qkv(layer.attention.query_key_value, layer_idx)
            register_forward_hook_dense_attention(layer.attention.dense, layer_idx)
            register_forward_hook_input_post_attention_layernorm(layer.post_attention_layernorm, layer_idx)
            register_forward_hook_mlp_h_to_4(layer.mlp.dense_h_to_4h, layer_idx)
            register_forward_hook_mlp_4_to_h(layer.mlp.dense_4h_to_h, layer_idx)
            register_forward_hook_output(layer, layer_idx)

    def _clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def untokenize_sample_tokenized(self, sample_idx):
        if sample_idx >= len(self.samples_untokenized):
            self.samples_untokenized.append([self.tokenizer.decode(token_idx) for token_idx in self.samples_tokenized[sample_idx].squeeze(0)])
        else:
            self.samples_untokenized[sample_idx] = [self.tokenizer.decode(token_idx) for token_idx in self.samples_tokenized[sample_idx].squeeze(0)]
        return self.samples_untokenized[sample_idx]
    
    def forward(self, sample_idx):
        self._register_hooks(sample_idx)

        with torch.no_grad():
            sample_tokenized = self.samples_tokenized[sample_idx].to(self.device)
            output = self.model(sample_tokenized, output_attentions=True)
        self._clear_hooks()

        logits, attentions = output["logits"], output["attentions"]
        if sample_idx >= len(self.logits):
            self.logits.append(logits)
            self.list_attentions.append(attentions)
        else:
            self.logits[sample_idx] = logits
            self.list_attentions[sample_idx] = attentions
    
    def get_histogram_activations_layer_token(self, sample_idx, type, layer, token, attn_head=None, rope_mode="full"):
        if type in ["q", "k", "v", "o", "o_mm_dense", "q_rope", "k_rope"]:
            attn_head = int(attn_head)

            dimension_start = attn_head*self.d_attn_head
            shape = (self.samples_tokenized[sample_idx].shape[1], self.d_model) if type != "o_mm_dense" else (self.samples_tokenized[sample_idx].shape[1], self.n_attn_heads, self.d_model)
            activations = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{layer}.bin", dtype=np.float32, shape=shape, mode="r")
            if type == "o_mm_dense":
                activations = activations[:, attn_head, :]
            else:
                activations = activations[:, dimension_start:dimension_start+self.d_attn_head]
            activations_median_histogram = self.activations_median_histogram[type][layer][attn_head]
            activations_bins_histogram = self.activations_bins_histogram[type][layer][attn_head]
            activations_bounds_histogram = self.activations_bounds_histogram[type][layer][attn_head]

            if type in ["q", "k", "q_rope", "k_rope"]:
                activations_median_histogram = activations_median_histogram[rope_mode]
                activations_bins_histogram = activations_bins_histogram[rope_mode]
                activations_bounds_histogram = activations_bounds_histogram[rope_mode]
                rope_dimensions = int(self.d_attn_head * self.rotary_pct)
                if rope_mode == "applied":
                    activations = activations[:, :rope_dimensions]

                elif rope_mode == "not_applied":
                    activations = activations[:, rope_dimensions:]

            if activations_bins_histogram == None:
                if activations_bounds_histogram == None:
                    activations_bounds_histogram = get_bounds_normalization(activations, quantiles=[0.25, 0.75], factors=[3, 3])
                
                n_bins = 30 if type != "o_mm_dense" else 40
                activations_bins_histogram = get_bins_histogram(activations_bounds_histogram[0], activations_bounds_histogram[1], n_bins)
                
            histogram_token = create_histogram(activations[token], activations_bins_histogram)

            if activations_median_histogram == None:
                activations_median_histogram = get_median_histogram_activations_layers(activations, activations_bins_histogram)
                
            outliers_idx, outliers_values = get_outliers(activations[token])

            if type in ["q", "k", "v", "o", "q_rope", "k_rope"]:
                outliers_idx = [outlier+self.d_attn_head*attn_head for outlier in outliers_idx]

            if type in ["q", "k", "q_rope", "k_rope"]:
                self.activations_median_histogram[type][layer][attn_head][rope_mode] = activations_median_histogram
                self.activations_bins_histogram[type][layer][attn_head][rope_mode] = activations_bins_histogram
                self.activations_bounds_histogram[type][layer][attn_head][rope_mode] = activations_bounds_histogram
            else:
                self.activations_median_histogram[type][layer][attn_head] = activations_median_histogram
                self.activations_bins_histogram[type][layer][attn_head] = activations_bins_histogram
                self.activations_bounds_histogram[type][layer][attn_head] = activations_bounds_histogram

            return {"bins_histogram": activations_bins_histogram, "histogram": histogram_token,
                    "median_histogram": activations_median_histogram, "outliers": {"name": outliers_idx, "values": outliers_values}}

        else:
            dimensionality = self.d_model * 4 if type == "mlp_h_to_4" else self.d_model
            activations = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{layer}.bin", dtype=np.float32, shape=(self.samples_tokenized[sample_idx].shape[1], dimensionality), mode="r")
            if self.activations_bins_histogram[type][layer] == None:
                if self.activations_bounds_histogram[type][layer] == None:
                    self.activations_bounds_histogram[type][layer] = get_bounds_normalization(activations, quantiles=[0.25, 0.75], factors=[3, 3])

                self.activations_bins_histogram[type][layer] = get_bins_histogram(self.activations_bounds_histogram[type][layer][0],
                                                                                    self.activations_bounds_histogram[type][layer][1], 40)
            
            histogram_token = create_histogram(activations[token], self.activations_bins_histogram[type][layer])
            
            if self.activations_median_histogram[type][layer] == None:
                self.activations_median_histogram[type][layer] = get_median_histogram_activations_layers(activations, self.activations_bins_histogram[type][layer])
            
            outliers_idx, outliers_values = get_outliers(activations[token])

            return {"bins_histogram": self.activations_bins_histogram[type][layer], "histogram": histogram_token,
                    "median_histogram": self.activations_median_histogram[type][layer], "outliers": {"name": outliers_idx, "values": outliers_values}}
    
    def get_attention(self, sample_idx, layer, attn_head, token):
        return self.list_attentions[sample_idx][layer].squeeze(0)[attn_head][token].tolist()
    
    def get_attention_opposite(self, sample_idx, layer, attn_head, token):
        return self.list_attentions[sample_idx][layer].squeeze(0)[attn_head][:, token].tolist()
    
    def get_losses(self, sample_idx):
        labels = self.samples_tokenized[sample_idx][:, 1:].to(self.device)
        losses = F.cross_entropy(self.logits[sample_idx][:, :-1, :].squeeze(0), labels.squeeze(0), reduction='none').tolist()

        return losses
    
    def get_dense_attention_sum(self, sample_idx, layer):
        dense_attention = np.memmap(f"{self.file_samples}/sample_{sample_idx}/dense_attention/dense_attention_layer_{layer}.bin", dtype=np.float32, shape=(self.samples_tokenized[sample_idx].shape[1], self.d_model), mode="r")
        dense_attention_sum = dense_attention.abs().sum(dim=1)
        dense_attention_sum = dense_attention_sum ** 2

        min_val = dense_attention_sum.min()
        max_val = dense_attention_sum.max()

        dense_attention_sum_normalized = (dense_attention_sum - min_val) / (max_val - min_val)

        return {"dense_attention_sum": dense_attention_sum.tolist(), "dense_attention_sum_normalized": dense_attention_sum_normalized.tolist()}
    
    def get_o_sum(self, sample_idx, layer):
        attn_head = 3
        o = np.memmap(f"{self.file_samples}/sample_{sample_idx}/o/o_layer_{layer}.bin", dtype=np.float32, shape=(self.samples_tokenized[sample_idx].shape[1], self.d_model), mode="r")[:, attn_head*self.d_attn_head:attn_head*self.d_attn_head+self.d_attn_head]
        o_sum = o.abs().sum(dim=1)
        o_sum = o_sum ** 2

        min_val = o_sum.min()
        max_val = o_sum.max()

        dense_attention_sum_normalized = (o_sum - min_val) / (max_val - min_val)

        return {"dense_attention_sum": o_sum.tolist(), "dense_attention_sum_normalized": dense_attention_sum_normalized.tolist()}
    
    def get_1668_value(self, sample_idx, layer):
        dense_attention = np.memmap(f"{self.file_samples}/sample_{sample_idx}/dense_attention/dense_attention_layer_{layer}.bin", dtype=np.float32, shape=(self.samples_tokenized[sample_idx].shape[1], self.d_model), mode="r")
        dense_attention_sum = abs(dense_attention).sum(axis=1)

        min_val_dense = 160 # 160
        max_val_dense = 1000 #1500

        dense_attention_sum_normalized = (dense_attention_sum - min_val_dense) / (max_val_dense - min_val_dense)

        #value_dim
        value_dim = -dense_attention[:, 1668]

        min_val_value_dim = -5
        max_val_value_dim = 5

        value_dim_normalized = (value_dim - min_val_value_dim) / (max_val_value_dim - min_val_value_dim)

        final_value = ((value_dim_normalized + dense_attention_sum_normalized) / 2) ** 2

        return {"dense_attention_sum": final_value.tolist(), "dense_attention_sum_normalized": final_value.tolist()}
    
    def get_top_predictions(self, sample_idx, top=5):
        logits_softmaxed = F.softmax(self.logits[sample_idx].squeeze(0), dim=1)
            
        values, indices = torch.topk(logits_softmaxed, top, dim=1)

        indices_list = [[self.tokenizer.decode(token_idx) for token_idx in row]for row in indices]
        values_list = values.tolist()

        return {"tokens": indices_list, "values": values_list}
    
    def get_similarities_tokens(self, sample_idx, type, layer, attn_head, token, rope_mode="full"):
        if type in ["q", "k", "v", "o", "o_mm_dense", "q_rope", "k_rope"]:
            attn_head = int(attn_head)
            
            # if type in ["q", "k", "q_rope", "k_rope"]:
            #     similarities_token = self.similarities_token[type][layer][attn_head][rope_mode]
            # else:
            #     similarities_token = self.similarities_token[type][layer][attn_head]

            # if similarities_token is None:
            dimension_start = attn_head*self.d_attn_head
            shape = (self.samples_tokenized[sample_idx].shape[1], self.d_model) if type != "o_mm_dense" else (self.samples_tokenized[sample_idx].shape[1], self.n_attn_heads, self.d_model)
            activations = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{layer}.bin", dtype=np.float32, shape=shape, mode="r")
            if type == "o_mm_dense":
                activations = activations[:, attn_head, :]
            else:
                activations = activations[:, dimension_start:dimension_start+self.d_attn_head]
            
            if type in ["q", "k", "q_rope", "k_rope"]:
                rope_dimensions = int(self.d_attn_head * self.rotary_pct)
                if rope_mode == "applied":
                    activations = activations[:, :rope_dimensions]

                elif rope_mode == "not_applied":
                    activations = activations[:, rope_dimensions:]

            norms = np.linalg.norm(activations, axis=1)
            norms = np.where(norms == 0, 1, norms)
            activations_norm = activations / norms[:, np.newaxis]
            cosine_similarity = np.dot(activations_norm, activations_norm.T)

            if type in ["q", "k", "q_rope", "k_rope"]:
                self.similarities_token[type][layer][attn_head][rope_mode] = cosine_similarity
            else:
                self.similarities_token[type][layer][attn_head] = cosine_similarity

            if type in ["q", "k", "q_rope", "k_rope"]:
                similarities = np.array(self.similarities_token[type][layer][attn_head][rope_mode][token])
            else:
                similarities = np.array(self.similarities_token[type][layer][attn_head][token])

        else:
            dimensionality = self.d_model * 4 if type == "mlp_h_to_4" else self.d_model
            activations = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{layer}.bin", dtype=np.float32, shape=(self.samples_tokenized[sample_idx].shape[1], dimensionality), mode="r")
            norms = np.linalg.norm(activations, axis=1)
            norms = np.where(norms == 0, 1, norms)
            activations_norm = activations / norms[:, np.newaxis]
            cosine_similarity = np.dot(activations_norm, activations_norm.T)

            self.similarities_token[type][layer] = cosine_similarity

            similarities = np.array(self.similarities_token[type][layer][token])

        min_val = similarities.min()
        max_val = similarities.max()

        similarities_normalized = (similarities - min_val) / (max_val - min_val)

        p1 = np.percentile(similarities, 1)
        p99 = np.percentile(similarities, 99)

        filtered_numbers = similarities[(similarities >= p1) & (similarities <= p99)]

        min_val = filtered_numbers.min()
        max_val = filtered_numbers.max()
        similarities_normalized_no_outliers = (similarities - min_val) / (max_val - min_val)

        return {"similarities": similarities.tolist(), "similarities_normalized": similarities_normalized.tolist(),
                    "similarities_normalized_no_outliers": similarities_normalized_no_outliers.tolist()}
        
    def get_similarities_previous(self, sample_idx, type, layer, attn_head, rope_mode="full"):
        if type in ["q", "k", "v", "o", "o_mm_dense", "q_rope", "k_rope"]:
            attn_head = int(attn_head)

            dimension_start = attn_head*self.d_attn_head

            shape = (self.samples_tokenized[sample_idx].shape[1], self.d_model) if type != "o_mm_dense" else (self.samples_tokenized[sample_idx].shape[1], self.n_attn_heads, self.d_model)
            activations = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{layer}.bin", dtype=np.float32, shape=shape, mode="r")
            activations_previous = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{layer-1}.bin", dtype=np.float32, shape=shape, mode="r")
            if type == "o_mm_dense":
                activations = activations[:, attn_head, :]
                activations_previous = activations_previous[:, attn_head, :]
            else:
                activations = activations[:, dimension_start:dimension_start+self.d_attn_head]
                activations_previous = activations_previous[:, dimension_start:dimension_start+self.d_attn_head]

            if type in ["q", "k", "q_rope", "k_rope"]:
                rope_dimensions = int(self.d_attn_head * self.rotary_pct)
                if rope_mode == "applied":
                    activations = activations[:, :rope_dimensions]
                    activations_previous = activations_previous[:, :rope_dimensions]

                elif rope_mode == "not_applied":
                    activations = activations[:, rope_dimensions:]
                    activations_previous = activations_previous[:, rope_dimensions:]

            norms = np.linalg.norm(activations, axis=1)
            norms = np.where(norms == 0, 1, norms)
            activations_norm = activations / norms[:, np.newaxis]

            norms_previous = np.linalg.norm(activations_previous, axis=1)
            norms_previous = np.where(norms_previous == 0, 1, norms_previous)
            activations_previous_norm = activations_previous / norms_previous[:, np.newaxis]

            cosine_similarity = (activations_norm * activations_previous_norm).sum(axis=1)

            if type in ["q", "k", "q_rope", "k_rope"]:
                self.similarities_previous[type][layer][attn_head][rope_mode] = cosine_similarity
            else:
                self.similarities_previous[type][layer][attn_head] = cosine_similarity

            if type in ["q", "k", "q_rope", "k_rope"]:
                similarities = np.array(self.similarities_previous[type][layer][attn_head][rope_mode])
            else:
                similarities = np.array(self.similarities_previous[type][layer][attn_head])

        else:
            dimensionality = self.d_model * 4 if type == "mlp_h_to_4" else self.d_model
            activations = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{layer}.bin", dtype=np.float32, shape=(self.samples_tokenized[sample_idx].shape[1], dimensionality), mode="r")
            previous_layer = int(layer) - 1
            activations_previous = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{previous_layer}.bin", dtype=np.float32, shape=(self.samples_tokenized[sample_idx].shape[1], dimensionality), mode="r")

            norms = np.linalg.norm(activations, axis=1)
            norms = np.where(norms == 0, 1, norms)
            activations_norm = activations / norms[:, np.newaxis]

            norms_previous = np.linalg.norm(activations_previous, axis=1)
            norms_previous = np.where(norms_previous == 0, 1, norms_previous)
            activations_previous_norm = activations_previous / norms_previous[:, np.newaxis]

            cosine_similarity = (activations_norm * activations_previous_norm).sum(axis=1)
            
            self.similarities_previous[type][layer] = cosine_similarity

            similarities = np.array(self.similarities_previous[type][layer])

        min_val = similarities.min()
        max_val = similarities.max()

        similarities_normalized = (similarities - min_val) / (max_val - min_val)

        p1 = np.percentile(similarities, 1)
        p99 = np.percentile(similarities, 99)

        filtered_numbers = similarities[(similarities >= p1) & (similarities <= p99)]

        min_val = filtered_numbers.min()
        max_val = filtered_numbers.max()
        similarities_normalized_no_outliers = (similarities - min_val) / (max_val - min_val)

        return {"similarities": similarities.tolist(), "similarities_normalized": similarities_normalized.tolist(),
                    "similarities_normalized_no_outliers": similarities_normalized_no_outliers.tolist()}
    
    def get_similarities_previous_residual(self, sample_idx, type, layer):
        activations = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{layer}.bin", dtype=np.float32, shape=((self.samples_tokenized[sample_idx].shape[1], self.d_model)), mode="r")
        activations_previous_residual = np.memmap(f"{self.file_samples}/sample_{sample_idx}/input/input_layer_{layer-1}.bin", dtype=np.float32, shape=((self.samples_tokenized[sample_idx].shape[1], self.d_model)), mode="r")
        if type in ["mlp_4_to_h", "output"]:
            activations_previous_residual = np.memmap(f"{self.file_samples}/sample_{sample_idx}/dense_attention_residual/dense_attention_residual_layer_{layer-1}.bin", dtype=np.float32, shape=((self.samples_tokenized[sample_idx].shape[1], self.d_model)), mode="r")

        norms = np.linalg.norm(activations, axis=1)
        norms = np.where(norms == 0, 1, norms)
        activations_norm = activations / norms[:, np.newaxis]

        norms_previous = np.linalg.norm(activations_previous_residual, axis=1)
        norms_previous = np.where(norms_previous == 0, 1, norms_previous)
        activations_previous_norm = activations_previous_residual / norms_previous[:, np.newaxis]

        similarities = (activations_norm * activations_previous_norm).sum(axis=1)

        min_val = similarities.min()
        max_val = similarities.max()

        similarities_normalized = (similarities - min_val) / (max_val - min_val)

        p1 = np.percentile(similarities, 1)
        p99 = np.percentile(similarities, 99)

        filtered_numbers = similarities[(similarities >= p1) & (similarities <= p99)]

        min_val = filtered_numbers.min()
        max_val = filtered_numbers.max()
        similarities_normalized_no_outliers = (similarities - min_val) / (max_val - min_val)

        return {"similarities": similarities.tolist(), "similarities_normalized": similarities_normalized.tolist(),
                    "similarities_normalized_no_outliers": similarities_normalized_no_outliers.tolist()}
    
    def get_activations_values(self, sample_idx, type, layer, attn_head, activation_idx):
        if type in ["q", "k", "v", "o", "o_mm_dense", "q_rope", "k_rope"]:
            attn_head = int(attn_head)

            shape = (self.samples_tokenized[sample_idx].shape[1], self.d_model) if type != "o_mm_dense" else (self.samples_tokenized[sample_idx].shape[1], self.n_attn_heads, self.d_model)
            activations = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{layer}.bin", dtype=np.float32, shape=shape, mode="r")

            if type != "o_mm_dense":
                dimension_start = attn_head*self.d_attn_head
                activations = activations[:, dimension_start:dimension_start+self.d_attn_head]
            else:
                activations = activations[:, attn_head, :]

        else:
            dimensionality = self.d_model * 4 if type == "mlp_h_to_4" else self.d_model
            activations = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{layer}.bin", dtype=np.float32, shape=(self.samples_tokenized[sample_idx].shape[1], dimensionality), mode="r")
        activations_values = activations[:, activation_idx]

        positives = activations_values[activations_values >= 0]
        negatives = activations_values[activations_values < 0]

        def normalize(values, min_val, max_val, is_negative=False):
            if is_negative:
                return (values - max_val) / (min_val - max_val) - 1
            else:
                return (values - min_val) / (max_val - min_val)

        if positives.size > 0 and negatives.size > 0:
            pos_max = positives.max()
            if positives.size > 1:
                pos_min = positives.min()
            else:
                pos_min = 0
            neg_min = negatives.min()
            if negatives.size > 1:
                neg_max = negatives.max()
            else:
                neg_max = 0
            normalized_positives = normalize(positives, pos_min, pos_max)
            normalized_negatives = normalize(negatives, neg_min, neg_max, is_negative=True)
        elif positives.size > 0:
            pos_min, pos_max = positives.min(), positives.max()
            normalized_positives = normalize(positives, pos_min, pos_max)
            normalized_negatives = np.array([])
        elif negatives.size > 0:
            neg_min, neg_max = negatives.min(), negatives.max()
            normalized_negatives = normalize(negatives, neg_min, neg_max, is_negative=True)
            normalized_positives = np.array([])

        activations_values_normalized = np.copy(activations_values)
        if positives.size > 0:
            activations_values_normalized[activations_values >= 0] = normalized_positives
        if negatives.size > 0:
            activations_values_normalized[activations_values < 0] = normalized_negatives

        def normalize_no_outliers(values, p1, p99, is_negative=False):
            filtered_values = values[(values >= p1) & (values <= p99)]
            if filtered_values.size == 0:
                filtered_values = values
            min_val = min(filtered_values.min(), -1e-10)
            max_val = max(filtered_values.max(), 1e-10)
            if is_negative:
                return (values - max_val) / (min_val - max_val) - 1
            else:
                return (values - min_val) / (max_val - min_val)

        activations_values_normalized_no_outliers = np.copy(activations_values)
        if positives.size > 0:
            if positives.size > 1:
                p1_pos, p99_pos = np.percentile(positives, 1), np.percentile(positives, 99)
            else:
                p1_pos, p99_pos = 0, positives.max()
            normalized_no_outliers_positives = normalize_no_outliers(positives, p1_pos, p99_pos)
            activations_values_normalized_no_outliers[activations_values >= 0] = normalized_no_outliers_positives

        if negatives.size > 0:
            if negatives.size > 1:
                p1_neg, p99_neg = np.percentile(negatives, 1), np.percentile(negatives, 99)
            else:
                p1_neg, p99_neg = negatives.min(), 0
            normalized_no_outliers_negatives = normalize_no_outliers(negatives, p1_neg, p99_neg, is_negative=True)
            activations_values_normalized_no_outliers[activations_values < 0] = normalized_no_outliers_negatives

        return {"activations": activations_values.tolist(), "activations_normalized": activations_values_normalized.tolist(),
                 "activations_normalized_no_outliers": activations_values_normalized_no_outliers.tolist()}

    def get_activations_sum(self, sample_idx, type, layer, attn_head):
        if type in ["q", "k", "v", "o", "o_mm_dense", "q_rope", "k_rope"]:
            attn_head = int(attn_head)

            shape = (self.samples_tokenized[sample_idx].shape[1], self.d_model) if type != "o_mm_dense" else (self.samples_tokenized[sample_idx].shape[1], self.n_attn_heads, self.d_model)
            activations = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{layer}.bin", dtype=np.float32, shape=shape, mode="r")

            if type != "o_mm_dense":
                dimension_start = attn_head*self.d_attn_head
                activations = activations[:, dimension_start:dimension_start+self.d_attn_head]
            else:
                activations = activations[:, attn_head, :]
        else:
            dimensionality = self.d_model * 4 if type == "mlp_h_to_4" else self.d_model
            activations = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{layer}.bin", dtype=np.float32, shape=(self.samples_tokenized[sample_idx].shape[1], dimensionality), mode="r")
        activations_sum = np.abs(activations).sum(axis=1)

        min_val = activations_sum.min()
        max_val = activations_sum.max()

        activations_sum_normalized = (activations_sum - min_val) / (max_val - min_val)

        p1 = np.percentile(activations_sum, 1)
        p99 = np.percentile(activations_sum, 99)

        filtered_numbers = activations_sum[(activations_sum >= p1) & (activations_sum <= p99)]

        min_val = filtered_numbers.min()
        max_val = filtered_numbers.max()
        activations_sum_normalized_no_outliers = (activations_sum - min_val) / (max_val - min_val)

        return {"activations_sum": activations_sum.tolist(), "activations_sum_normalized": activations_sum_normalized.tolist(),
                 "activations_sum_normalized_no_outliers": activations_sum_normalized_no_outliers.tolist()}
    
    def get_activation_value_histogram(self, sample_idx, dim, type, layer, token, attn_head=None):
        if type in ["q", "k", "v", "o", "o_mm_dense", "q_rope", "k_rope"]:
            attn_head = int(attn_head)
            shape = (self.samples_tokenized[sample_idx].shape[1], self.d_model) if type != "o_mm_dense" else (self.samples_tokenized[sample_idx].shape[1], self.n_attn_heads, self.d_model)
            activations = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{layer}.bin", dtype=np.float32, shape=shape, mode="r")
            if type == "o_mm_dense":
                activation_value = activations[token, attn_head, dim]
            else:
                dimension_start = attn_head*self.d_attn_head
                dim_to_check = dim - dimension_start
                if dim_to_check < 0:
                    return
                activation_value = activations[token, dim]

        else:
            activation_value = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{layer}.bin", dtype=np.float32, shape=(self.samples_tokenized[sample_idx].shape[1], self.d_model), mode="r")[token, dim]

        return [activation_value.item()]
    
    def get_activation_vector(self, sample_idx, type, layer, attn_head, token_idx):
        if type in ["q", "k", "v", "o", "o_mm_dense", "q_rope", "k_rope"]:
            attn_head = int(attn_head)

            shape = (self.samples_tokenized[sample_idx].shape[1], self.d_model) if type != "o_mm_dense" else (self.samples_tokenized[sample_idx].shape[1], self.n_attn_heads, self.d_model)
            activations = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{layer}.bin", dtype=np.float32, shape=shape, mode="r")

            if type != "o_mm_dense":
                dimension_start = attn_head*self.d_attn_head
                activations = activations[:, dimension_start:dimension_start+self.d_attn_head]
        else:
            dimensionality = self.d_model * 4 if type == "mlp_h_to_4" else self.d_model
            activations = np.memmap(f"{self.file_samples}/sample_{sample_idx}/{type}/{type}_layer_{layer}.bin", dtype=np.float32, shape=(self.samples_tokenized[sample_idx].shape[1], dimensionality), mode="r")

        activations_vector = activations[token_idx]

        return activations_vector.tolist()