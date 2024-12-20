from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn.functional as F
import sentencepiece as spm
from torch import Tensor, device
from pathlib import Path
from layers import TamilGPT

@dataclass
class GenerationConfig:
    """Configuration for text generation parameters"""
    max_length: int = 100
    temperature: float = 0.7
    top_k: int = 50
    context_length: int = 256

class TamilTextGenerator:
    """A class for generating Tamil text using a pre-trained TamilGPT model."""
    
    def __init__(
        self, 
        model_path: str | Path,
        tokenizer_path: str | Path,
        device_name: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        """
        Initialize the text generator with model and tokenizer.
        
        Args:
            model_path: Path to the saved model checkpoint
            tokenizer_path: Path to the SentencePiece tokenizer model
            device_name: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = torch.device(device_name)
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.model = self._load_model(model_path)
        
    def _load_tokenizer(self, tokenizer_path: str | Path) -> spm.SentencePieceProcessor:
        """Load the SentencePiece tokenizer."""
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(str(tokenizer_path))
        return tokenizer
    
    def _load_model(self, model_path: str | Path) -> torch.nn.Module:
        """Load the TamilGPT model from checkpoint."""
        model = TamilGPT(
            vocab_size=32000,
            embedding_dimension=768,
            context_length=256,
            num_heads=2,
            scaling_factor=2,
            num_layers=2,
            bias=False,
            dropout=0,
            weight_tying=True
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def _prepare_input(self, text: str) -> Tensor:
        """Convert input text to tensor."""
        input_ids = self.tokenizer.encode(text)
        return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def _sample_next_token(
        self,
        logits: Tensor,
        temperature: float,
        top_k: int
    ) -> Tensor:
        """Sample the next token using top-k sampling."""
        scaled_logits = logits / temperature
        top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k) # top k
        probabilities = F.softmax(top_k_logits, dim=-1)
        next_token_index = torch.multinomial(probabilities, 1)
        return top_k_indices[next_token_index]
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate text based on input prompt.
        
        Args:
            prompt: Initial text to start generation
            config: Generation configuration parameters
            
        Returns:
            Generated text including the prompt
        """
        if config is None:
            config = GenerationConfig()
            
        input_tensor = self._prepare_input(prompt)
        
        for _ in range(config.max_length):
            input_tensor = input_tensor[:, -config.context_length:]
            
            logits = self.model(input_tensor)
            last_token_logits = logits[0, -1, :]
            
            next_token = self._sample_next_token(
                last_token_logits,
                config.temperature,
                config.top_k
            )
            
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
            
        generated_ids = [int(token) for token in input_tensor[0].cpu().numpy()]
        return self.tokenizer.decode(generated_ids)


if __name__ == "__main__":
    generator = TamilTextGenerator(
        model_path='checkpoints/tamilgpt_epoch1_loss10.9288.pth',
        tokenizer_path='models/tok32000.model'
    )
    
    config = GenerationConfig(
        max_length=10,
        temperature=0.3,
        top_k=50
    )
    
    text = generator.generate("வணக்கம்", config)
    print("Generated Text:", text)