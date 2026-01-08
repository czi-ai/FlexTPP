import torch

class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        is_causal=True,
        add_bias_kv=False,
        add_zero_attn=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.is_causal = is_causal

        # Compute head dimension. Must divide embed_dim exactly.
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Define projection layers for query, key, and value.
        # Here we project from embed_dim to embed_dim so that after splitting
        # into num_heads we have dimension head_dim for each head.
        self.q = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v = torch.nn.Linear(embed_dim, embed_dim, bias=bias)

        # Optionally add bias parameters for key and value.
        if add_bias_kv:
            # Each has shape (1, 1, embed_dim) and will be expanded along the batch dimension.
            self.bias_k = torch.nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = torch.nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k, self.bias_v = None, None

        # Output projection maps from embed_dim back to embed_dim.
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = torch.nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize weights following the xavier_uniform initialization.
        torch.nn.init.xavier_uniform_(self.q.weight)
        torch.nn.init.xavier_uniform_(self.k.weight)
        torch.nn.init.xavier_uniform_(self.v.weight)
        torch.nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q.bias is not None:
            torch.nn.init.constant_(self.q.bias, 0.)
            torch.nn.init.constant_(self.k.bias, 0.)
            torch.nn.init.constant_(self.v.bias, 0.)
            torch.nn.init.constant_(self.out_proj.bias, 0.)
        if self.add_bias_kv:
            torch.nn.init.xavier_normal_(self.bias_k)
            torch.nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value):
        """
        query, key, value: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = query.size()

        # Project inputs to multi-head Q, K, V.
        q = self.q(query)  # (batch_size, seq_len, embed_dim)
        k = self.k(key)
        v = self.v(value)

        # If using additional bias for key and value, append them along the sequence dimension.
        if self.add_bias_kv:
            # Expand bias parameters to (batch_size, 1, embed_dim)
            bias_k = self.bias_k.expand(batch_size, -1, -1)
            bias_v = self.bias_v.expand(batch_size, -1, -1)
            k = torch.cat([k, bias_k], dim=1)  # new seq_len = seq_len + 1
            v = torch.cat([v, bias_v], dim=1)

        # Reshape and transpose for multi-head attention.
        # New shape: (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # For k and v, sequence length may be increased if add_bias_kv is True.
        new_seq_len = k.size(1)
        k = k.view(batch_size, new_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, new_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention using the built-in function.
        # Note: torch.nn.functional.scaled_dot_product_attention expects queries, keys with matching head_dim.
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, is_causal=self.is_causal
        )

        # Optionally, if add_zero_attn is True, append an all-zero vector to the output.
        if self.add_zero_attn:
            zero_attn = torch.zeros(batch_size, self.num_heads, 1, self.head_dim, device=attn_output.device, dtype=attn_output.dtype)
            attn_output = torch.cat([attn_output, zero_attn], dim=2)

        # Reshape back to (batch_size, seq_len, embed_dim)
        # If add_zero_attn is True, we remove the extra token after the output projection.
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        if self.add_zero_attn:
            attn_output = attn_output[:, :seq_len, :]

        # Apply the final output projection.
        attn_output = self.out_proj(attn_output)
        attn_output = self.attn_dropout(attn_output)
        return attn_output
