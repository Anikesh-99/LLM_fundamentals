import numpy as np

def stable_softmax(x):
    e_x = np.exp(x - np.max(x, axis = -1, keepdims=True))
    return e_x / np.sum(e_x, axis = -1, keepdims = True)

def normal_softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis = -1, keepdims = True)

def cross_entropy(logits, correct):
    m = np.max(logits, keepdims=True, axis = -1)
    lse = m + np.log(np.sum(np.exp(logits - m), keepdims=True, axis = -1))
    return lse - logits[correct]

print(stable_softmax(np.array([1000, 1, 10, -1])))
print(normal_softmax(np.array([1000, 1, 10, -1])))
print(cross_entropy(np.array([1000, 1, 10, -1]), 2))


