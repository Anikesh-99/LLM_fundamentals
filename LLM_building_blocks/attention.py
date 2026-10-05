import numpy as np

# 12288 x 12288 for Wv, 12288 x 128 for Wq and Wk 
embedding_size = 12288
h = 96
dk = embedding_size
rng = np.random.default_rng()
Wq, Wk, Wv = rng.normal(size=(embedding_size, embedding_size)), rng.normal(size=(embedding_size, embedding_size)), rng.normal(size=(embedding_size, embedding_size))
# Wo = rng.normal((embedding_size, embedding_size))

def softmax(x, temperature=1):
    e_x = np.exp((x - np.max(x, axis = -1, keepdims=True)))
    return e_x / np.sum(e_x, axis = -1, keepdims = True)

# single head
def scaled_dot_product_attention(X):
    print(X.shape, Wq.shape)
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    print(Q.shape, K.shape, "Shapes of Query and Key")
    print(V.shape, "Shape of Value")
    scores = Q@K.T/np.sqrt(dk)
    print(scores.shape, "Shape of score")
    mask = np.triu(np.ones((seq, seq), bool), k = 1)
    scores[mask] = -np.inf
    score_softmax = softmax(scores)
    print(score_softmax.shape, 'Shape of softmax')
    output = score_softmax @ V
    return output


# 10 token length sequence
seq = 10
X = rng.random((seq, embedding_size))
print(scaled_dot_product_attention(X).shape)

# def temperature_sampling(logits, temperature):
#     scores = softmax(logits, temperature)
#     return scores

# def top_k_sampling(logits, k):
#     idx = np.argpartition(logits, -k, axis=0)[-k:]
#     mask = np.ones(logits.shape, bool)
#     mask[idx] = False
#     logits[mask] = -np.inf
#     scores = softmax(logits)
#     return scores

# def top_p_sampling(logits, p):
#     original_indices = np.argsort(logits)
#     scores = softmax(logits)
#     curr_p = 0
#     idx = original_indices.size - 1
#     while idx >= 0:
#         curr_p += scores[original_indices[idx]]
#         if curr_p > p: break
#         idx -= 1
#     idx -= 1
#     while idx >= 0:
#         scores[original_indices[idx]] = 0
#         idx -= 1
#     return scores/scores.sum()


