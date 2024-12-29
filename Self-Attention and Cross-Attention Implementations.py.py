import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBase(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Base attention class implementing core attention functionality
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the last dimension into (n_heads, d_k)"""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge the head dimensions back"""
        batch_size, n_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)


class CausalAttention(AttentionBase):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Causal (masked) self-attention implementation
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__(d_model, n_heads, dropout)

    def forward(self, 
                q: torch.Tensor,
                k: torch.Tensor, 
                v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for causal attention
        
        Args:
            q: Query tensor of shape (batch_size, seq_len, d_model)
            k: Key tensor of shape (batch_size, seq_len, d_model)
            v: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = q.size(0), q.size(1)

        # Create causal mask if not provided
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.to(q.device)

        # Linear projections and split heads
        q = self.split_heads(self.W_q(q))  # (batch_size, n_heads, seq_len, d_k)
        k = self.split_heads(self.W_k(k))  # (batch_size, n_heads, seq_len, d_k)
        v = self.split_heads(self.W_v(v))  # (batch_size, n_heads, seq_len, d_k)

        # Scaled dot-product attention with masking
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)  # (batch_size, n_heads, seq_len, seq_len)
        
        # Apply causal mask
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        out = torch.matmul(attn_weights, v)  # (batch_size, n_heads, seq_len, d_k)
        
        # Merge heads and apply output projection
        out = self.merge_heads(out)  # (batch_size, seq_len, d_model)
        out = self.W_o(out)
        
        return out


class NonCausalAttention(AttentionBase):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Non-causal (bidirectional) self-attention implementation
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__(d_model, n_heads, dropout)

    def forward(self, 
                q: torch.Tensor,
                k: torch.Tensor, 
                v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for non-causal attention
        
        Args:
            q: Query tensor of shape (batch_size, seq_len, d_model)
            k: Key tensor of shape (batch_size, seq_len, d_model)
            v: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional padding mask tensor
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Linear projections and split heads
        q = self.split_heads(self.W_q(q))  # (batch_size, n_heads, seq_len, d_k)
        k = self.split_heads(self.W_k(k))  # (batch_size, n_heads, seq_len, d_k)
        v = self.split_heads(self.W_v(v))  # (batch_size, n_heads, seq_len, d_k)

        # Scaled dot-product attention
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)  # (batch_size, n_heads, seq_len, seq_len)
        
        # Apply padding mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        out = torch.matmul(attn_weights, v)  # (batch_size, n_heads, seq_len, d_k)
        
        # Merge heads and apply output projection
        out = self.merge_heads(out)  # (batch_size, seq_len, d_model)
        out = self.W_o(out)
        
        return out


# Example usage
def example_usage():
    # Model parameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize both attention variants
    causal_attn = CausalAttention(d_model, n_heads)
    non_causal_attn = NonCausalAttention(d_model, n_heads)
    
    # Forward pass through both attention mechanisms
    causal_output = causal_attn(x, x, x)  # Self-attention
    non_causal_output = non_causal_attn(x, x, x)  # Self-attention
    
    print(f"Causal attention output shape: {causal_output.shape}")
    print(f"Non-causal attention output shape: {non_causal_output.shape}")

if __name__ == "__main__":
    example_usage()