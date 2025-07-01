import numpy as np
from datasets import load_dataset
import os

# Sigmoid helper function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load and preprocess SST2 dataset
dataset = load_dataset("glue", "sst2")
train_data = dataset["train"]
test_data = dataset["validation"]

# Build vocabulary from training data
word_set = set()
for example in train_data:
    for word in example["sentence"].lower().split():
        word_set.add(word)
vocab = sorted(list(word_set))
vocab_size = len(vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

# Convert sentences to indices and pad
def sentence_to_indices(sentence, word_to_ix, max_len):
    indices = [word_to_ix[w] for w in sentence.lower().split() if w in word_to_ix]
    padded = indices + [-1] * (max_len - len(indices))
    return padded[:max_len]

max_len = 20
X_train_indices = np.array([sentence_to_indices(ex["sentence"], word_to_ix, max_len) for ex in train_data])
Y_train = np.array([ex["label"] for ex in train_data])
X_test_indices = np.array([sentence_to_indices(ex["sentence"], word_to_ix, max_len) for ex in test_data])
Y_test = np.array([ex["label"] for ex in test_data])

def initialize_parameters(n_a, n_x, n_y):
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x)*0.01
    Waa = np.random.randn(n_a, n_a)*0.01
    Wya = np.random.randn(n_y, n_a)*0.01
    ba = np.zeros((n_a, 1))
    by = np.zeros((n_y, 1))
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
    return parameters

def initialize_embedding(vocab_size, emb_dim):
    np.random.seed(1)
    Wex = np.random.randn(emb_dim, vocab_size) * 0.01
    return Wex

def rnn_forward_manual_emb(x_emb_list, a0, parameters):
    caches = []
    n_a, m = a0.shape
    T_x = len(x_emb_list)
    n_y, _ = parameters["Wya"].shape
    a = np.zeros((n_a, m, T_x))
    a_next = a0
    for t in range(T_x):
        xt_emb = x_emb_list[t]
        a_prev = a_next
        Waa, Wax, Wya, ba, by = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['ba'], parameters['by']
        a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt_emb) + ba)
        a[:, :, t] = a_next
        cache = (a_next, a_prev, xt_emb, parameters)
        caches.append(cache)
    y_pred = sigmoid(np.dot(Wya, a_next) + by)
    return a, y_pred, caches

def compute_cost(y_pred, Y, parameters=None, lambd=0.0):
    m = Y.shape[0]
    cross_entropy_cost = -1/m * np.sum(Y * np.log(y_pred + 1e-8) + (1-Y) * np.log(1-y_pred + 1e-8))
    reg_cost = 0
    if parameters is not None and lambd > 0:
        reg_cost += np.sum(parameters['Wax'] ** 2)
        reg_cost += np.sum(parameters['Waa'] ** 2)
        reg_cost += np.sum(parameters['Wya'] ** 2)
        cross_entropy_cost += (lambd/(2*m)) * reg_cost
    return np.squeeze(cross_entropy_cost)

def rnn_backward(y_pred, Y, caches, x_indices, Wex, lambd=0.0):
    (caches_cell, x_emb_caches) = caches
    (a_next, a_prev, xt_emb, parameters) = caches_cell[-1]
    m = Y.shape[0]
    T_x = len(caches_cell)
    n_a, _ = a_next.shape
    emb_dim, _ = xt_emb.shape
    vocab_size = Wex.shape[1]

    dWax = np.zeros_like(parameters["Wax"])
    dWaa = np.zeros_like(parameters["Waa"])
    dWya = np.zeros_like(parameters["Wya"])
    dba = np.zeros_like(parameters["ba"])
    dby = np.zeros_like(parameters["by"])
    da_prevt = np.zeros((n_a, m))
    dWex = np.zeros((emb_dim, vocab_size))

    dZ = y_pred - Y.reshape(1, -1)
    dWya = (1/m) * np.dot(dZ, a_next.T)
    dby = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    da_next = np.dot(parameters['Wya'].T, dZ)

    for t in reversed(range(T_x)):
        a_next, a_prev, xt_emb, parameters = caches_cell[t]
        dtanh = (1 - a_next**2) * da_next
        dxt_emb = np.dot(parameters['Wax'].T, dtanh)
        dWax += np.dot(dtanh, xt_emb.T)
        da_prevt = np.dot(parameters['Waa'].T, dtanh)
        dWaa += np.dot(dtanh, a_prev.T)
        dba += np.sum(dtanh, axis=1, keepdims=True)
        xt_indices = x_indices[:, t]
        for i in range(m):
            idx = xt_indices[i]
            if idx != -1:
                dWex[:, idx] += dxt_emb[:, i].reshape(-1)
        da_next = da_prevt

    if lambd > 0:
        dWax += (lambd/m) * parameters['Wax']
        dWaa += (lambd/m) * parameters['Waa']
        dWya += (lambd/m) * parameters['Wya']

    gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "dba": dba, "dby": dby, "da0": da_prevt, "dWex": dWex}
    return gradients

def update_parameters(parameters, gradients, learning_rate, Wex, clip_threshold=1.0):
    # Gradient clipping
    for grad in ['dWax', 'dWaa', 'dWya', 'dba', 'dby', 'dWex']:
        if np.linalg.norm(gradients[grad]) > clip_threshold:
            gradients[grad] = gradients[grad] * (clip_threshold / np.linalg.norm(gradients[grad]))

    parameters['Wax'] -= learning_rate * gradients['dWax']
    parameters['Waa'] -= learning_rate * gradients['dWaa']
    parameters['Wya'] -= learning_rate * gradients['dWya']
    parameters['ba'] -= learning_rate * gradients['dba']
    parameters['by'] -= learning_rate * gradients['dby']
    Wex -= learning_rate * gradients['dWex']
    return parameters, Wex

def model(X, Y, n_a, emb_dim, vocab_size, num_iterations=1000, learning_rate=0.05, lambd=0.0):
    np.random.seed(1)
    m, T_x = X.shape
    n_y = 1
    parameters = initialize_parameters(n_a, emb_dim, n_y)
    Wex = initialize_embedding(vocab_size, emb_dim)
    for i in range(num_iterations):
        a0 = np.zeros((n_a, m))
        x_emb_caches = []
        for t in range(T_x):
            xt_indices = X[:, t]
            xt = np.zeros((vocab_size, m))
            for j in range(m):
                if xt_indices[j] != -1:
                    xt[xt_indices[j], j] = 1
            x_emb_caches.append(np.dot(Wex, xt))
        a, y_pred, caches_cell = rnn_forward_manual_emb(x_emb_caches, a0, parameters)
        caches = (caches_cell, x_emb_caches)
        cost = compute_cost(y_pred, Y, parameters, lambd)
        gradients = rnn_backward(y_pred, Y, caches, X, Wex, lambd)
        parameters, Wex = update_parameters(parameters, gradients, learning_rate, Wex)
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost:.4f}")
    return parameters, Wex

def predict(sentence, parameters, Wex, word_to_ix, max_len):
    words = sentence.lower().split()
    indices = [word_to_ix[w] for w in words if w in word_to_ix]
    if not indices:
        return "Cannot predict (no words from vocab found)."
    padded_indices = np.array(indices + [-1] * (max_len - len(indices))).reshape(1, -1)
    n_a, _ = parameters['Waa'].shape
    a0 = np.zeros((n_a, 1))
    x_emb_list = []
    for t in range(padded_indices.shape[1]):
        idx = padded_indices[0, t]
        if idx != -1:
            x_emb = Wex[:, idx].reshape(-1, 1)
        else:
            x_emb = np.zeros((Wex.shape[0], 1))
        x_emb_list.append(x_emb)
    _, y_pred, _ = rnn_forward_manual_emb(x_emb_list, a0, parameters)
    prediction = (y_pred > 0.5).astype(int)
    sentiment = "Positive" if prediction.item() == 1 else "Negative"
    return f'Sentiment: {sentiment} (Score: {y_pred.item():.4f})'

subset = 100  # Using a small training subset
hidden_layer_size = 16
embedding_dimension = 16
lambd = 0.01
trained_parameters, trained_Wex = model(
    X_train_indices[:subset], Y_train[:subset], hidden_layer_size, embedding_dimension, vocab_size,
    num_iterations=1000, learning_rate=0.05, lambd=lambd
)

# Test predictions
for i in range(5):
    sent = test_data[i]["sentence"]
    print(f'Sentence: "{sent}" -> {predict(sent, trained_parameters, trained_Wex, word_to_ix, max_len)}')