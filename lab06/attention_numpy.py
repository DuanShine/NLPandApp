import numpy as np

def stable_softmax(x, axis=-1, temperature=1.0):
    x_scaled = x / temperature
    x_max = np.max(x_scaled, axis=axis, keepdims=True)
    exp_x = np.exp(x_scaled - x_max)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp_x

def scaled_dot_product_attention(Q, K, V, mask=None, temperature=1.0):
    d_k = K.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)
    
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
        
    attention_weights = stable_softmax(scores, axis=-1, temperature=temperature)
    context = np.matmul(attention_weights, V)
    
    return scores, attention_weights, context

class SelfAttentionNumPy:
    def __init__(self, d_model, d_k, random_seed=42):
        self.d_model = d_model
        self.d_k = d_k
        if random_seed is not None:
            np.random.seed(random_seed)
            
        limit = 1.0 / np.sqrt(d_model)
        self.W_Q = np.random.uniform(-limit, limit, (d_model, d_k)).astype(np.float32)
        self.W_K = np.random.uniform(-limit, limit, (d_model, d_k)).astype(np.float32)
        self.W_V = np.random.uniform(-limit, limit, (d_model, d_k)).astype(np.float32)

    def forward(self, X, mask=None, temperature=1.0):
        Q = np.matmul(X, self.W_Q)
        K = np.matmul(X, self.W_K)
        V = np.matmul(X, self.W_V)
        
        scores, attention_weights, context = scaled_dot_product_attention(
            Q, K, V, mask=mask, temperature=temperature
        )
        return scores, attention_weights, context

if __name__ == "__main__":
    q_test = np.array([[1.0, 0.0]], dtype=np.float32)
    k_test = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    v_test = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    
    scores, weights, ctx = scaled_dot_product_attention(q_test, k_test, v_test, temperature=1.0)
    print("Scores:\n", scores)
    print("Attention Weights:\n", weights)
    print("Context Vector:\n", ctx)
    
    overflow_test = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
    print("Stable Softmax Overflow Test:", stable_softmax(overflow_test))
