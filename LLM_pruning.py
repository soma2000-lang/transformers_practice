import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm
import logging
import os
import json
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PruningConfig:
    """Configuration for model pruning"""
    model_name: str = "meta-llama/Llama-2-7b-hf"
    pruning_type: str = "width"  # Options: "width", "depth", "2:4"
    pruning_ratio: float = 0.3  # Percentage of components to prune
    attention_head_importance_metric: str = "l1_norm"  # Options: "l1_norm", "attention_scores"
    evaluation_batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "pruned_model"
    evaluation_steps: int = 100

class AttentionPruner:
    """Handles pruning of attention heads in transformer models"""
    
    def __init__(self, model: LlamaForCausalLM, config: PruningConfig):
        self.model = model
        self.config = config
        self.head_importance = {}
        
    def compute_head_importance(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores for each attention head using specified metric
        """
        logger.info(f"Computing head importance using {self.config.attention_head_importance_metric}")
        self.model.eval()
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Get attention module
            attention = layer.self_attn
            num_heads = attention.num_heads
            
            if self.config.attention_head_importance_metric == "l1_norm":
                # Compute L1 norm of attention weights
                query_weight = attention.q_proj.weight.view(num_heads, -1)
                key_weight = attention.k_proj.weight.view(num_heads, -1)
                value_weight = attention.v_proj.weight.view(num_heads, -1)
                
                head_importance = (
                    torch.norm(query_weight, p=1, dim=1) +
                    torch.norm(key_weight, p=1, dim=1) +
                    torch.norm(value_weight, p=1, dim=1)
                )
                
            elif self.config.attention_head_importance_metric == "attention_scores":
                head_importance = torch.zeros(num_heads, device=self.config.device)
                
                with torch.no_grad():
                    for batch in tqdm(dataloader, desc=f"Computing attention scores for layer {layer_idx}"):
                        inputs = batch["input_ids"].to(self.config.device)
                        attention_mask = batch["attention_mask"].to(self.config.device)
                        
                        # Get attention scores
                        outputs = self.model(
                            input_ids=inputs,
                            attention_mask=attention_mask,
                            output_attentions=True
                        )
                        
                        # Aggregate attention scores
                        attention_scores = outputs.attentions[layer_idx]
                        head_importance += attention_scores.mean(dim=[0, 1, 2])
                
                head_importance /= len(dataloader)
            
            self.head_importance[f"layer_{layer_idx}"] = head_importance
            
        return self.head_importance

    def prune_heads(self) -> None:
        """
        Prune attention heads based on importance scores
        """
        num_heads_to_prune = int(self.model.config.num_attention_heads * self.config.pruning_ratio)
        logger.info(f"Pruning {num_heads_to_prune} heads per layer")
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Get importance scores for this layer
            importance = self.head_importance[f"layer_{layer_idx}"]
            
            # Find heads to prune
            _, indices = torch.topk(importance, 
                                  k=num_heads_to_prune, 
                                  largest=False)
            
            # Create head mask
            head_mask = torch.ones(self.model.config.num_attention_heads, 
                                 device=self.config.device)
            head_mask[indices] = 0
            
            # Apply mask to attention weights
            attention = layer.self_attn
            
            # Mask query, key, value projections
            attention.q_proj.weight.data *= head_mask.view(-1, 1)
            attention.k_proj.weight.data *= head_mask.view(-1, 1)
            attention.v_proj.weight.data *= head_mask.view(-1, 1)
            
            logger.info(f"Pruned {num_heads_to_prune} heads in layer {layer_idx}")

class DepthPruner:
    """Handles pruning of transformer layers"""
    
    def __init__(self, model: LlamaForCausalLM, config: PruningConfig):
        self.model = model
        self.config = config
        self.layer_importance = {}
    
    def compute_layer_importance(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Compute importance scores for each layer using gradient-based metrics
        """
        logger.info("Computing layer importance")
        self.model.eval()
        
        # Register hooks to get gradients
        gradients = {}
        def save_gradient(name):
            def hook(grad):
                gradients[name] = grad.detach()
            return hook
        
        # Register hooks for each layer
        handles = []
        for idx, layer in enumerate(self.model.model.layers):
            handle = layer.register_backward_hook(save_gradient(f"layer_{idx}"))
            handles.append(handle)
        
        # Compute gradients
        for batch in tqdm(dataloader, desc="Computing layer importance"):
            inputs = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            
            outputs = self.model(
                input_ids=inputs,
                attention_mask=attention_mask,
                labels=inputs
            )
            
            loss = outputs.loss
            loss.backward()
            
            # Aggregate importance scores
            for name, gradient in gradients.items():
                if name not in self.layer_importance:
                    self.layer_importance[name] = 0
                self.layer_importance[name] += torch.norm(gradient).item()
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Normalize importance scores
        total_importance = sum(self.layer_importance.values())
        for key in self.layer_importance:
            self.layer_importance[key] /= total_importance
            
        return self.layer_importance
    
    def prune_layers(self) -> None:
        """
        Prune transformer layers based on importance scores
        """
        num_layers_to_prune = int(len(self.model.model.layers) * self.config.pruning_ratio)
        logger.info(f"Pruning {num_layers_to_prune} layers")
        
        # Sort layers by importance
        sorted_layers = sorted(
            self.layer_importance.items(),
            key=lambda x: x[1]
        )
        
        # Get indices of layers to remove
        layers_to_remove = [
            int(layer[0].split('_')[1])
            for layer in sorted_layers[:num_layers_to_prune]
        ]
        
        # Create new layer list excluding pruned layers
        new_layers = nn.ModuleList([
            layer
            for idx, layer in enumerate(self.model.model.layers)
            if idx not in layers_to_remove
        ])
        
        # Replace model layers
        self.model.model.layers = new_layers
        logger.info(f"Pruned {num_layers_to_prune} layers")

class TwoFourPruner:
    """Implements 2:4 structured pruning"""
    
    def __init__(self, model: LlamaForCausalLM, config: PruningConfig):
        self.model = model
        self.config = config
    
    def apply_2_4_sparsity(self) -> None:
        """
        Apply 2:4 structured sparsity to linear layers
        """
        logger.info("Applying 2:4 structured sparsity")
        
        def prune_2_4(tensor: torch.Tensor) -> torch.Tensor:
            """
            Prune 2 out of every 4 weights, keeping the largest 2
            """
            # Reshape tensor to group weights into chunks of 4
            original_shape = tensor.shape
            reshaped = tensor.view(-1, 4)
            
            # Get indices of smallest 2 weights in each group
            _, indices = torch.topk(torch.abs(reshaped), k=2, dim=1, largest=False)
            
            # Create mask
            mask = torch.ones_like(reshaped)
            mask.scatter_(1, indices, 0)
            
            # Apply mask and reshape back
            pruned = reshaped * mask
            return pruned.view(original_shape)
        
        # Apply 2:4 sparsity to all linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = prune_2_4(module.weight.data)
                logger.info(f"Applied 2:4 sparsity to {name}")

class ModelPruner:
    """Main class for model pruning"""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        
        # Load model and tokenizer
        logger.info(f"Loading model {config.model_name}")
        self.model = LlamaForCausalLM.from_pretrained(config.model_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
        
        # Move model to device
        self.model.to(config.device)
        
        # Initialize pruners
        self.attention_pruner = AttentionPruner(self.model, config)
        self.depth_pruner = DepthPruner(self.model, config)
        self.two_four_pruner = TwoFourPruner(self.model, config)
    
    def evaluate_model(self, dataloader: DataLoader) -> float:
        """
        Evaluate model perplexity
        """
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                
                outputs = self.model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    labels=inputs
                )
                
                total_loss += outputs.loss.item()
                total_steps += 1
                
                if total_steps >= self.config.evaluation_steps:
                    break
        
        return torch.exp(torch.tensor(total_loss / total_steps))
    
    def prune_model(self, dataloader: DataLoader) -> None:
        """
        Apply specified pruning technique
        """
        logger.info(f"Starting {self.config.pruning_type} pruning")
        
        # Evaluate before pruning
        initial_perplexity = self.evaluate_model(dataloader)
        logger.info(f"Initial perplexity: {initial_perplexity:.2f}")
        
        if self.config.pruning_type == "width":
            self.attention_pruner.compute_head_importance(dataloader)
            self.attention_pruner.prune_heads()
            
        elif self.config.pruning_type == "depth":
            self.depth_pruner.compute_layer_importance(dataloader)
            self.depth_pruner.prune_layers()
            
        elif self.config.pruning_type == "2:4":
            self.two_four_pruner.apply_2_4_sparsity()
        
        # Evaluate after pruning
        final_perplexity = self.evaluate_model(dataloader)
        logger.info(f"Final perplexity: {final_perplexity:.2f}")
        
        # Save pruned model
        self.save_model()
    
    def save_model(self) -> None:
        """
        Save pruned model and configuration
        """
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)
            
        # Save model
        self.model.save_pretrained(self.config.save_path)
        self.tokenizer.save_pretrained(self.config.save_path)
        
        # Save pruning configuration
        config_path = os.path.join(self.config.save_path, "pruning_config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2)
            
        logger.info(f"Saved pruned model to {self.config.save_path}")

class TextDataset(Dataset):
    """Simple dataset for evaluation"""
    
    def __init__(self, texts: List[str], tokenizer: LlamaTokenizer, max_length: int = 512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            key: val[idx]
            for key, val in self.encodings.items()
        }
    
    def __len__(self) -> int:
        return len(self.encodings.input_ids)

def main():
    # Example usage
    config = PruningConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        pruning_type="width",
        pruning_ratio=0.3,
        save_path="pruned_llama"
    )
    
    
    # Initialize pruner
    pruner = ModelPruner(config)
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, pruner.tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.evaluation_batch_size)
    
    # Prune model
    pruner.prune_model(dataloader)

if __name__ == "__main__":
    main()
