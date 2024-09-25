import numpy as np
import fasttext
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Step 1: Data Preparation
def load_fasttext_embeddings(file_path, vocab_size=100000):
    model = fasttext.load_model(file_path)
    words = []
    vectors = []
    for word in tqdm(model.get_words()[:vocab_size]):
        words.append(word)
        vectors.append(model[word])
    return {word: vector for word, vector in zip(words, vectors)}

def load_bilingual_lexicon(file_path):
    lexicon = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            src, tgt = line.strip().split()
            lexicon[src] = tgt
    return lexicon

# Step 2: Embedding Alignment
def align_embeddings(src_emb, tgt_emb, lexicon):
    src_words = list(lexicon.keys())
    tgt_words = list(lexicon.values())
    
    src_matrix = np.array([src_emb[word] for word in src_words if word in src_emb])
    tgt_matrix = np.array([tgt_emb[word] for word in tgt_words if word in tgt_emb])
    
    # Ensure matrices have the same number of rows
    min_rows = min(src_matrix.shape[0], tgt_matrix.shape[0])
    src_matrix = src_matrix[:min_rows]
    tgt_matrix = tgt_matrix[:min_rows]
    
    # Compute the orthogonal Procrustes solution
    W, _ = orthogonal_procrustes(src_matrix, tgt_matrix)
    return W

# Step 3: Evaluation
def word_translation(src_word, src_emb, tgt_emb, W, k=5):
    if src_word not in src_emb:
        return []
    
    query_vec = np.dot(src_emb[src_word], W)
    similarities = cosine_similarity(query_vec.reshape(1, -1), list(tgt_emb.values()))
    top_k_indices = similarities.argsort()[0][-k:][::-1]
    return [list(tgt_emb.keys())[idx] for idx in top_k_indices]

def evaluate_translation(test_dict, src_emb, tgt_emb, W):
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    for src_word, tgt_word in test_dict.items():
        if src_word not in src_emb or tgt_word not in tgt_emb:
            continue
        
        translations = word_translation(src_word, src_emb, tgt_emb, W)
        if translations[0] == tgt_word:
            correct_1 += 1
        if tgt_word in translations:
            correct_5 += 1
        total += 1
    
    p1 = correct_1 / total
    p5 = correct_5 / total
    return p1, p5

def main():
    # Load embeddings
    print("Loading embeddings...")
    en_emb = load_fasttext_embeddings('path/to/english_embeddings.bin')
    hi_emb = load_fasttext_embeddings('path/to/hindi_embeddings.bin')
    
    # Load bilingual lexicon
    print("Loading bilingual lexicon...")
    train_lexicon = load_bilingual_lexicon('path/to/muse_train_lexicon.txt')
    test_lexicon = load_bilingual_lexicon('path/to/muse_test_lexicon.txt')
    
    # Align embeddings
    print("Aligning embeddings...")
    W = align_embeddings(en_emb, hi_emb, train_lexicon)
    
    # Evaluate
    print("Evaluating...")
    p1, p5 = evaluate_translation(test_lexicon, en_emb, hi_emb, W)
    print(f"Precision@1: {p1:.4f}")
    print(f"Precision@5: {p5:.4f}")
    
    # Ablation study
    print("Performing ablation study...")
    sizes = [5000, 10000, 20000]
    for size in sizes:
        small_lexicon = dict(list(train_lexicon.items())[:size])
        W = align_embeddings(en_emb, hi_emb, small_lexicon)
        p1, p5 = evaluate_translation(test_lexicon, en_emb, hi_emb, W)
        print(f"Lexicon size: {size}")
        print(f"Precision@1: {p1:.4f}")
        print(f"Precision@5: {p5:.4f}")

if __name__ == "__main__":
    main()
