import torch
import torch.nn as nn
import types

class Adapter(nn.Module):
    def __init__(self, emb_dim, adapter_dim=256, dropout=0.2):
        """
        Args:
            emb_dim (int): Hidden dimension of the backbone model.
            adapter_dim (int): Bottleneck dimension of the adapter.
            dropout (float): Dropout rate applied inside the adapter.
        """
        super().__init__()
        self.down_proj = nn.Linear(emb_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the adapter.

        Args:
            x (Tensor): Input hidden states of shape [B, L, emb_dim].

        Returns:
            Tensor: Residual-enhanced hidden states.
        """
        z = self.down_proj(x)
        z = self.activation(z)
        z = self.dropout(z)
        z = self.up_proj(z)
        return x + z  # Residual connection


def add_adapters_to_t5(model, adapter_dim=256, dropout=0.2):
    """
    Add adapter modules to all encoder and decoder blocks of a T5 model.

    Args:
        model: HuggingFace T5 model instance.
        adapter_dim (int): Adapter bottleneck dimension.
        dropout (float): Adapter dropout rate.
    """
    # Attach adapter modules
    for block in model.encoder.block:
        block.adapter = Adapter(model.config.d_model, adapter_dim, dropout)

    for block in model.decoder.block:
        block.adapter = Adapter(model.config.d_model, adapter_dim, dropout)

    # Cache original forward methods
    orig_encoder_forward = model.encoder.block[0].__class__.forward
    orig_decoder_forward = model.decoder.block[0].__class__.forward

    # Define new forward with adapter
    def encoder_block_forward_with_adapter(self, *args, **kwargs):
        outputs = orig_encoder_forward(self, *args, **kwargs)
        hidden_states = outputs[0]
        hidden_states = self.adapter(hidden_states)
        return (hidden_states,) + outputs[1:]

    def decoder_block_forward_with_adapter(self, *args, **kwargs):
        outputs = orig_decoder_forward(self, *args, **kwargs)
        hidden_states = outputs[0]
        hidden_states = self.adapter(hidden_states)
        return (hidden_states,) + outputs[1:]

    # Monkey-patch forward functions
    for block in model.encoder.block:
        block.forward = types.MethodType(encoder_block_forward_with_adapter, block)

    for block in model.decoder.block:
        block.forward = types.MethodType(decoder_block_forward_with_adapter, block)

def freeze_except_three_layers_and_adapters(model):
    """
    Freeze all model parameters except:
        - First 3 encoder blocks
        - First 2 decoder blocks + last decoder block
        - All adapter parameters
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze selected encoder blocks
    for block in model.encoder.block[:3]:
        for param in block.parameters():
            param.requires_grad = True

    # Unfreeze selected decoder blocks
    for block in model.decoder.block[:2]:
        for param in block.parameters():
            param.requires_grad = True

    for param in model.decoder.block[-1].parameters():
        param.requires_grad = True

    # Always train adapters
    for block in model.encoder.block:
        for param in block.adapter.parameters():
            param.requires_grad = True

    for block in model.decoder.block:
        for param in block.adapter.parameters():
            param.requires_grad = True


class ConditionalMolT5(nn.Module):
    """
    Conditional MolT5 supporting multiple conditioning modes and
    optional post-hoc cross-attention analysis.
    """

    def __init__(
        self,
        model,
        tokenizer,
        cond_dim=128,
        cond_tokens=32,
        cond_mode="vector",  # "vector" | "token_emb"
    ):
        """
        Args
        ----
        model:
            Pretrained T5-based MolT5 model.
        tokenizer:
            Tokenizer for SMILES decoding.
        cond_dim : int
            Dimension of condition vector (used in vector mode).
        cond_tokens : int
            Number of virtual condition tokens (vector mode).
        cond_mode : str
            Conditioning mode:
                - "vector": continuous condition vector
                - "token_emb": token-level condition embeddings
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        self.cond_dim = cond_dim
        self.cond_tokens = cond_tokens
        self.cond_mode = cond_mode

        # Projection only used for vector-based conditioning
        if self.cond_mode == "vector":
            self.cond_proj = nn.Linear(
                cond_dim,
                self.model.config.d_model * cond_tokens
            )

        if self.cond_mode == "token_emb":
            self.cond_proj = nn.Linear(
                cond_dim,
                self.model.config.d_model
            )

    def _encode_condition(self, cond_input):
        """
        Encode condition input into encoder embeddings.

        Args
        ----
        cond_input:
            - vector mode: Tensor [B, cond_dim]
            - token_emb mode: Tensor [B, Lc, d_model]

        Returns
        -------
        cond_emb : Tensor [B, Lc, d_model]
        cond_attention_mask : Tensor [B, Lc]
        """
        device = cond_input.device

        if self.cond_mode == "vector":
            batch_size = cond_input.size(0)
            cond_emb = self.cond_proj(cond_input).view(
                batch_size, self.cond_tokens, -1
            )
            cond_attention_mask = torch.ones(
                batch_size, self.cond_tokens, device=device
            )

        elif self.cond_mode == "token_emb":
            cond_emb = self.cond_proj(cond_input)
            cond_attention_mask = torch.ones(
                cond_emb.size(0), cond_emb.size(1), device=device
            )

        else:
            raise ValueError(f"Unknown cond_mode: {self.cond_mode}")

        return cond_emb, cond_attention_mask

    def forward(
        self,
        cond_input,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None
    ):
        """
        Forward pass with conditional embeddings injected
        into the encoder input.
        """
        cond_emb, cond_attention_mask = self._encode_condition(cond_input)

        if labels is not None:
            labels = labels.contiguous()

        return self.model(
            inputs_embeds=cond_emb,
            attention_mask=cond_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def generate(self, cond_input, max_length=200, **generate_kwargs):
        """
        Conditional sequence generation.
        """
        cond_emb, cond_attention_mask = self._encode_condition(cond_input)

        return self.model.generate(
            inputs_embeds=cond_emb,
            attention_mask=cond_attention_mask,
            max_length=max_length,
            **generate_kwargs
        )

    def generate_smiles(self, cond_input, max_length=200, **generate_kwargs):
        """
        Generate SMILES strings conditioned on input.
        """
        generated_ids = self.generate(
            cond_input,
            max_length=max_length,
            **generate_kwargs
        )
        return [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in generated_ids
        ]

    def analyze_cross_attention(
        self,
        cond_input,
        decoder_input_ids,
        decoder_attention_mask=None,
    ):
        """
        Analyze decoder cross-attention to condition tokens.

        This method is ONLY valid when cond_mode == "token_emb".
        It does NOT affect training or generation.

        Args
        ----
        cond_input : Tensor
            Condition token embeddings [B, Lc, d_model].
        decoder_input_ids : LongTensor
            Decoder input token ids [B, Lt].
        decoder_attention_mask : Tensor, optional
            Decoder attention mask [B, Lt].

        Returns
        -------
        List[Tensor]
            Per-layer maximum cross-attention values.
            Each tensor has shape [B].
        """
        if self.cond_mode != "token_emb":
            raise RuntimeError(
                "Cross-attention analysis is only supported in cond_mode='token_emb'."
            )

        self.model.eval()

        with torch.no_grad():
            cond_emb, cond_attention_mask = self._encode_condition(cond_input)

            outputs = self.model(
                inputs_embeds=cond_emb,
                attention_mask=cond_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=True,
                return_dict=True,
            )

        return self._extract_decoder_cross_attn_max(outputs)

    @staticmethod
    def _extract_decoder_cross_attn_max(outputs):
        """
        Extract maximum cross-attention per decoder layer.

        Args
        ----
        outputs:
            ModelOutput with cross_attentions enabled.

        Returns
        -------
        List[Tensor]
            Each element is [B], one per decoder layer.
        """
        if outputs.cross_attentions is None:
            raise ValueError("No cross-attention found in model outputs.")

        layerwise_max = []

        for layer_attn in outputs.cross_attentions:
            # [B, num_heads, tgt_len, cond_len]
            max_attn = layer_attn.max(dim=-1)[0]   # cond_len
            max_attn = max_attn.max(dim=-1)[0]     # tgt_len
            max_attn = max_attn.max(dim=-1)[0]     # heads
            layerwise_max.append(max_attn)         # [B]

        return layerwise_max
