import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Existing functions from the previous implementation
# (load_fasttext_embeddings, load_bilingual_lexicon, evaluate_translation)
# ... (include these functions here) ...

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def csls_sim(x, y, k=10):
    sim = cosine_similarity(x, y)
    sx = np.sort(sim, axis=1)[:, -k-1:-1].mean(axis=1)[:, np.newaxis]
    sy = np.sort(sim.T, axis=1)[:, -k-1:-1].mean(axis=1)[:, np.newaxis]
    return 2 * sim - sx - sy.T

def unsupervised_alignment(src_emb, tgt_emb, n_epochs=5, batch_size=32, disc_hidden=2048):
    src_words = list(src_emb.keys())
    tgt_words = list(tgt_emb.keys())
    
    src_vectors = torch.FloatTensor(np.array([src_emb[w] for w in src_words]))
    tgt_vectors = torch.FloatTensor(np.array([tgt_emb[w] for w in tgt_words]))
    
    input_dim = src_vectors.shape[1]
    
    # Initialize W as an identity matrix
    W = torch.eye(input_dim, requires_grad=True)
    
    # Initialize discriminator
    discriminator = Discriminator(input_dim, disc_hidden)
    
    # Optimizers
    w_optimizer = optim.SGD([W], lr=0.1)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.1)
    
    # Loss function
    criterion = nn.BCELoss()
    
    for epoch in range(n_epochs):
        # Train discriminator
        for _ in range(5):
            d_optimizer.zero_grad()
            
            # Sample batch
            src_idx = torch.randint(0, len(src_words), (batch_size,))
            tgt_idx = torch.randint(0, len(tgt_words), (batch_size,))
            
            src_batch = src_vectors[src_idx]
            tgt_batch = tgt_vectors[tgt_idx]
            
            # Transform source embeddings
            src_trans = torch.mm(src_batch, W)
            
            # Discriminator predictions
            src_pred = discriminator(src_trans)
            tgt_pred = discriminator(tgt_batch)
            
            # Compute loss
            src_loss = criterion(src_pred, torch.zeros(batch_size, 1))
            tgt_loss = criterion(tgt_pred, torch.ones(batch_size, 1))
            d_loss = (src_loss + tgt_loss) / 2
            
            d_loss.backward()
            d_optimizer.step()
        
        # Train mapping
        w_optimizer.zero_grad()
        
        src_idx = torch.randint(0, len(src_words), (batch_size,))
        src_batch = src_vectors[src_idx]
        
        src_trans = torch.mm(src_batch, W)
        src_pred = discriminator(src_trans)
        
        w_loss = criterion(src_pred, torch.ones(batch_size, 1))
        
        w_loss.backward()
        w_optimizer.step()
        
        # Orthogonalize W
        W.data = torch.from_numpy(np.linalg.svd(W.data.numpy())[0].dot(np.linalg.svd(W.data.numpy())[2]))
        
        print(f"Epoch {epoch+1}/{n_epochs}, D Loss: {d_loss.item():.4f}, W Loss: {w_loss.item():.4f}")
    
    return W.detach().numpy()

def main():
    # Load embeddings and lexicons (as in the previous implementation)
    # ...

    # Supervised Procrustes alignment
    print("Performing supervised Procrustes alignment...")
    W_supervised = align_embeddings(en_emb, hi_emb, train_lexicon)
    p1_supervised, p5_supervised = evaluate_translation(test_lexicon, en_emb, hi_emb, W_supervised)
    print(f"Supervised - Precision@1: {p1_supervised:.4f}, Precision@5: {p5_supervised:.4f}")

    # Unsupervised alignment
    print("Performing unsupervised alignment with adversarial training and CSLS...")
    W_unsupervised = unsupervised_alignment(en_emb, hi_emb)
    p1_unsupervised, p5_unsupervised = evaluate_translation(test_lexicon, en_emb, hi_emb, W_unsupervised)
    print(f"Unsupervised - Precision@1: {p1_unsupervised:.4f}, Precision@5: {p5_unsupervised:.4f}")

    # Compare results
    print("\nComparison:")
    print(f"Supervised   - Precision@1: {p1_supervised:.4f}, Precision@5: {p5_supervised:.4f}")
    print(f"Unsupervised - Precision@1: {p1_unsupervised:.4f}, Precision@5: {p5_unsupervised:.4f}")

if __name__ == "__main__":
    main()
