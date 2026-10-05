import numpy as np
import time
embedding_size = 12288
dk = embedding_size
rng = np.random.default_rng()
Wq, Wk, Wv = rng.normal(size=(embedding_size, embedding_size)), rng.normal(size=(embedding_size, embedding_size)), rng.normal(size=(embedding_size, embedding_size))

def softmax(x, temperature=1):
    e_x = np.exp((x - np.max(x, axis = -1, keepdims=True))/temperature)
    return e_x / np.sum(e_x, axis = -1, keepdims = True)

def scaled_dot_product_attention(X, new_token = None):
    X = np.concatenate((X, new_token), axis = 0)
    ln = X.shape[0]
    # unoptimized without caching
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    scores = Q@K.T/np.sqrt(dk)
    mask = np.triu(np.ones((ln, ln), bool), k = 1)
    scores[mask] = -np.inf
    score_softmax = softmax(scores)
    output = score_softmax @ V
    return output

K_cache, V_cache = None, None
def cached_scaled_dot_product_attention(X, new_token = None):
    global K_cache, V_cache
    ln = X.shape[0]
    if K_cache is None or V_cache is None:
        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv
        K_cache = K
        V_cache = V
        scores = Q@K.T/np.sqrt(dk)
        mask = np.triu(np.ones((ln, ln), bool), k = 1)
        scores[mask] = -np.inf
        score_softmax = softmax(scores)
        output = score_softmax @ V
    else:
        q_new = new_token @ Wq
        k_new = new_token @ Wk
        v_new = new_token @ Wv
        K_cache = np.concatenate([K_cache, k_new], axis = 0)
        V_cache = np.concatenate([V_cache, v_new], axis = 0)
        scores = q_new @ K_cache.T/np.sqrt(dk)
        weights = softmax(scores)
        output = weights @ V_cache
    return output

seq = 10
X = rng.random((seq, embedding_size))
start_time = time.perf_counter()
for i in range(100):
    new_token = rng.random((1, embedding_size))
    new_score = scaled_dot_product_attention(X, new_token)
    X = np.concatenate((X, new_token), axis = 0)
end_time = time.perf_counter()
print("Uncached took us: ", end_time - start_time)
X = rng.random((seq, embedding_size))
start_time = time.perf_counter()
for i in range(100):
    new_token = rng.random((1, embedding_size))
    new_score = cached_scaled_dot_product_attention(X, new_token)
    X = np.concatenate((X, new_token), axis = 0)
end_time = time.perf_counter()
print("Cached took us: ", end_time - start_time)