import numpy as np
import fasttext
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm

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

def load_fasttext_embeddings(language):
    """Load pre-trained FastText embeddings for the specified language."""
    if language == 'en':
        model = fasttext.load_model('cc.en.300.bin')
    elif language == 'hi':
        model = fasttext.load_model('cc.hi.300.bin')
    else:
        raise ValueError("Unsupported language")
    return model

def load_muse_dictionary(filename):
    """Load the MUSE bilingual dictionary."""
    word_pairs = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            src, tgt = line.strip().split()
            word_pairs.append((src, tgt))
    return word_pairs

def csls_similarity(x, y, k=10):
    """Compute CSLS similarity between x and y."""
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
    
    cos_sim = np.dot(x_norm, y_norm.T)
    
    x_mean_sim = np.mean(np.sort(cos_sim, axis=1)[:, -k:], axis=1)
    y_mean_sim = np.mean(np.sort(cos_sim.T, axis=1)[:, -k:], axis=1)
    
    csls_sim = 2 * cos_sim - x_mean_sim[:, np.newaxis] - y_mean_sim[np.newaxis, :]
    return csls_sim

def procrustes_align(X, Y):
    """Align matrices X and Y using Procrustes analysis."""
    U, _, Vt = np.linalg.svd(np.dot(Y.T, X))
    return np.dot(U, Vt)

def supervised_alignment(src_emb, tgt_emb, word_pairs):
    """Align embeddings using supervised Procrustes method."""
    src_words = [pair[0] for pair in word_pairs]
    tgt_words = [pair[1] for pair in word_pairs]
    
    X = np.array([src_emb[word] for word in src_words if word in src_emb])
    Y = np.array([tgt_emb[word] for word in tgt_words if word in tgt_emb])
    
    return procrustes_align(X, Y)

def unsupervised_alignment(src_emb, tgt_emb, num_iterations=5, batch_size=32, discriminator_iterations=5):
    """Align embeddings using unsupervised method with CSLS and adversarial training."""
    input_dim = src_emb.shape[1]
    hidden_dim = 2048
    
    W = torch.randn(input_dim, input_dim, requires_grad=True)
    W = nn.Parameter(torch.Tensor(procrustes_align(src_emb, tgt_emb)))
    
    discriminator = Discriminator(input_dim, hidden_dim)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.1)
    w_optimizer = optim.Adam([W], lr=0.1)
    
    for it in tqdm(range(num_iterations), desc="Unsupervised Alignment"):
        # Discriminator training
        for _ in range(discriminator_iterations):
            src_idx = np.random.randint(0, src_emb.shape[0], batch_size)
            tgt_idx = np.random.randint(0, tgt_emb.shape[0], batch_size)
            
            src_batch = torch.FloatTensor(src_emb[src_idx])
            tgt_batch = torch.FloatTensor(tgt_emb[tgt_idx])
            
            src_aligned = torch.mm(src_batch, W)
            
            d_optimizer.zero_grad()
            
            src_pred = discriminator(src_aligned)
            tgt_pred = discriminator(tgt_batch)
            
            d_loss = -torch.mean(torch.log(tgt_pred) + torch.log(1 - src_pred))
            d_loss.backward()
            d_optimizer.step()
        
        # Mapping training
        src_idx = np.random.randint(0, src_emb.shape[0], batch_size)
        src_batch = torch.FloatTensor(src_emb[src_idx])
        
        w_optimizer.zero_grad()
        
        src_aligned = torch.mm(src_batch, W)
        src_pred = discriminator(src_aligned)
        
        w_loss = -torch.mean(torch.log(src_pred))
        w_loss.backward()
        w_optimizer.step()
        
        # Orthogonalize W
        W.data = torch.Tensor(procrustes_align(W.data.numpy(), np.eye(input_dim)))
    
    return W.detach().numpy()

def evaluate_alignment(src_emb, tgt_emb, W, test_pairs, method='csls'):
    """Evaluate alignment using either CSLS or cosine similarity."""
    src_words = [pair[0] for pair in test_pairs]
    tgt_words = [pair[1] for pair in test_pairs]
    
    X = np.array([src_emb[word] for word in src_words if word in src_emb])
    Y = np.array([tgt_emb[word] for word in tgt_words if word in tgt_emb])
    
    X_aligned = np.dot(X, W)
    
    if method == 'csls':
        sim = csls_similarity(X_aligned, Y)
    else:  # cosine
        sim = cosine_similarity(X_aligned, Y)
    
    top1 = sim.argmax(axis=1)
    top5 = np.argsort(-sim, axis=1)[:, :5]
    
    precision1 = np.mean(top1 == np.arange(len(top1)))
    precision5 = np.mean([i in top5[j] for j, i in enumerate(range(len(top5)))])
    
    return precision1, precision5

def combined_alignment(src_emb, tgt_emb, train_pairs, num_iterations=5):
    """Combine supervised and unsupervised methods for alignment."""
    # Start with supervised alignment
    W_supervised = supervised_alignment(src_emb, tgt_emb, train_pairs)
    
    # Use supervised result as initialization for unsupervised method
    src_aligned = np.dot(src_emb, W_supervised)
    W_combined = unsupervised_alignment(src_aligned, tgt_emb, num_iterations=num_iterations)
    
    # Combine the two transformations
    return np.dot(W_supervised, W_combined)

def main():
    print("Loading embeddings...")
    en_model = load_fasttext_embeddings('en')
    hi_model = load_fasttext_embeddings('hi')
    
    print("Loading MUSE dictionary...")
    train_pairs = load_muse_dictionary('muse_en_hi_train.txt')
    test_pairs = load_muse_dictionary('muse_en_hi_test.txt')
    
    src_emb = en_model.get_input_matrix()
    tgt_emb = hi_model.get_input_matrix()
    
    print("Performing supervised alignment (Procrustes)...")
    W_supervised = supervised_alignment(en_model, hi_model, train_pairs)
    p1_sup, p5_sup = evaluate_alignment(en_model, hi_model, W_supervised, test_pairs, method='cosine')
    print(f"Supervised Results - P@1: {p1_sup:.4f}, P@5: {p5_sup:.4f}")
    
    print("\nPerforming unsupervised alignment (CSLS + Adversarial)...")
    W_unsupervised = unsupervised_alignment(src_emb, tgt_emb)
    p1_unsup, p5_unsup = evaluate_alignment(en_model, hi_model, W_unsupervised, test_pairs)
    print(f"Unsupervised Results - P@1: {p1_unsup:.4f}, P@5: {p5_unsup:.4f}")
    
    print("\nPerforming combined alignment...")
    W_combined = combined_alignment(src_emb, tgt_emb, train_pairs)
    p1_comb, p5_comb = evaluate_alignment(en_model, hi_model, W_combined, test_pairs)
    print(f"Combined Results - P@1: {p1_comb:.4f}, P@5: {p5_comb:.4f}")
    
    print("\nComparison:")
    print(f"Supervised   - P@1: {p1_sup:.4f}, P@5: {p5_sup:.4f}")
    print(f"Unsupervised - P@1: {p1_unsup:.4f}, P@5: {p5_unsup:.4f}")
    print(f"Combined     - P@1: {p1_comb:.4f}, P@5: {p5_comb:.4f}")

if __name__ == "__main__":
    main()
