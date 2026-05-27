#!/usr/bin/env python
# coding: utf-8

# # Coupled Input‑Forget Gate LSTM Implementation

# In[1]:


import numpy as np

class CIFG_LSTM:
    def __init__(self, input_size, hidden_size):
        """
        Initialize network parameters
        input_size: dimension of the input vector at each time step
        hidden_size: dimension of the hidden state (memory)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights with a normal distribution and biases with zeros
        # W: weights connected to input X
        # U: weights connected to previous hidden state h
        # b: biases
        
        # 1. Input Gate parameters
        self.Wi = np.random.randn(hidden_size, input_size) * 0.1
        self.Ui = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bi = np.zeros((hidden_size, 1))
        
        # 2. Candidate Cell State parameters
        self.Wc = np.random.randn(hidden_size, input_size) * 0.1
        self.Uc = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bc = np.zeros((hidden_size, 1))
        
        # 3. Output Gate parameters
        self.Wo = np.random.randn(hidden_size, input_size) * 0.1
        self.Uo = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bo = np.zeros((hidden_size, 1))

    def _sigmoid(self, x):
        """Sigmoid activation function producing output between 0 and 1"""
        # Prevent numerical overflow
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def _tanh(self, x):
        """Hyperbolic tangent activation function producing output between -1 and 1"""
        return np.tanh(x)

    def forward_step(self, x_t, h_prev, c_prev):
        """
        Perform the forward pass for a single time step
        x_t: input at time t with shape (input_size, 1)
        h_prev: previous hidden state with shape (hidden_size, 1)
        c_prev: previous cell state with shape (hidden_size, 1)
        """
        # 1. Compute input gate (i_t)
        i_t = self._sigmoid(np.dot(self.Wi, x_t) + np.dot(self.Ui, h_prev) + self.bi)
        
        # Forget gate is implicitly defined as (1 - i_t)
        f_t = 1.0 - i_t
        
        # 2. Compute candidate cell state (c_tilde_t)
        c_tilde_t = self._tanh(np.dot(self.Wc, x_t) + np.dot(self.Uc, h_prev) + self.bc)
        
        # 3. Update the cell state (C_t) using the given formula
        # Ct = (1 − it) ⊙ Ct−1 + it ⊙ C˜t
        c_t = f_t * c_prev + i_t * c_tilde_t  # In NumPy, * means element-wise multiplication
        
        # 4. Compute output gate (o_t)
        o_t = self._sigmoid(np.dot(self.Wo, x_t) + np.dot(self.Uo, h_prev) + self.bo)
        
        # 5. Compute the new hidden state (h_t)
        h_t = o_t * self._tanh(c_t)
        
        return h_t, c_t

    def forward(self, X):
        """
        Perform the forward pass for a full input sequence
        X: input sequence with shape (sequence_length, input_size)
        """
        seq_length = X.shape[0]
        
        # Initialize memory states with zeros
        h_t = np.zeros((self.hidden_size, 1))
        c_t = np.zeros((self.hidden_size, 1))
        
        # List to store all hidden states over time
        hidden_states = []
        
        # Loop through time steps
        for t in range(seq_length):
            # Reshape the t-th input into a column vector (input_size, 1)
            x_t = X[t].reshape(-1, 1)
            
            # Run one LSTM step
            h_t, c_t = self.forward_step(x_t, h_t, c_t)
            
            # Store output
            hidden_states.append(h_t.flatten())
            
        return np.array(hidden_states)


# In[3]:


import time
# ==========================================
# Testing the code on a Synthetic Time Series
# ==========================================
if __name__ == "__main__":
    # Dimension settings based on problem requirements
    INPUT_SIZE = 10   # Input vector dimension at each time step
    HIDDEN_SIZE = 16  # Hidden state (memory) dimension
    SEQ_LENGTH = 50   # Synthetic time series length (50 time steps)

    # 1. Generate synthetic time series randomly
    np.random.seed(42) # For reproducible results
    X_synthetic = np.random.randn(SEQ_LENGTH, INPUT_SIZE)

    # 2. Create model instance 
    # (Random weight initialization is done automatically in __init__)
    model = CIFG_LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)

    # 3. Run the model and get outputs
    all_hidden_states = model.forward(X_synthetic)

    # --- Verify matrix dimensions ---
    print("=== Matrix Dimensions Verification ===")
    print(f"Input time series shape: {X_synthetic.shape} -> (Seq_len=50, Input_size=10)")
    print(f"Total output matrix shape: {all_hidden_states.shape} -> (Seq_len=50, Hidden_size=16)")
    assert all_hidden_states.shape == (SEQ_LENGTH, HIDDEN_SIZE), "Output dimension mismatch error!"
    print("Matrix dimensions verified successfully.\n")

    # --- Extract h_t at each time step ---
    print("=== Extracting h_t at each time step ===")
    
    # Print the initial state (h_0)
    # Assuming initial state is zeros as per standard LSTM initialization
    h_0 = np.zeros(HIDDEN_SIZE)
    print(f"Time Step 00 (Initial h_0) | Shape: {h_0.shape}")
    print(f"Values: {np.round(h_0, 3)}\n")

    for t in range(SEQ_LENGTH):
        h_t = all_hidden_states[t] # Extract hidden state vector at time t
        
        # To avoid excessively long console output, print only the first few and the last step
        if t < 3:
            print(f"Time Step {t+1:02d} | Shape: {h_t.shape}")
            print(f"Values: {np.round(h_t, 3)}\n")
        elif t == 3:
            print("...\n[Outputs for steps 4 to 49 calculated and extracted, but hidden for brevity]\n...")
        elif t == SEQ_LENGTH - 1:
            print(f"Time Step {t+1:02d} (Last Step) | Shape: {h_t.shape}")
            print(f"Values: {np.round(h_t, 3)}\n")
    # ==========================================
    # Performance Benchmarking
    # ==========================================
    print("=== Performance Benchmarking ===")
    num_runs = 1000  # تعداد تکرار برای محاسبه دقیق‌تر زمان
    
    # گرم کردن پردازنده (Warm-up)
    for _ in range(10):
        _ = model.forward(X_synthetic)
        
    # شروع اندازه‌گیری زمان
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        _ = model.forward(X_synthetic)
        
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_per_run = (total_time / num_runs) * 1000 # بر حسب میلی‌ثانیه
    
    print(f"Total time for {num_runs} forward passes: {total_time:.4f} seconds")
    print(f"Average time per forward pass: {avg_time_per_run:.4f} milliseconds")
    print("========================================\n")


# # Optimized Coupled Input‑Forget Gate LSTM Implementation

# In[4]:


import numpy as np

class CIFG_LSTM:
    def __init__(self, input_size, hidden_size):
        """
        Initialize parameters (same interface as the original non‑optimized version)
        input_size: dimension of input vector
        hidden_size: dimension of hidden state
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Original-style initialization (same distributions as the first code)
        Wi = np.random.randn(hidden_size, input_size) * 0.1
        Ui = np.random.randn(hidden_size, hidden_size) * 0.1
        bi = np.zeros((hidden_size, 1))

        Wc = np.random.randn(hidden_size, input_size) * 0.1
        Uc = np.random.randn(hidden_size, hidden_size) * 0.1
        bc = np.zeros((hidden_size, 1))

        Wo = np.random.randn(hidden_size, input_size) * 0.1
        Uo = np.random.randn(hidden_size, hidden_size) * 0.1
        bo = np.zeros((hidden_size, 1))

        # -------- Optimized combined matrices --------
        # Shape: (input_size + hidden_size , 3 * hidden_size)
        self.W = np.zeros((input_size + hidden_size, 3 * hidden_size))

        # Input part
        self.W[:input_size, 0:hidden_size] = Wi.T
        self.W[:input_size, hidden_size:2*hidden_size] = Wc.T
        self.W[:input_size, 2*hidden_size:3*hidden_size] = Wo.T

        # Hidden part
        self.W[input_size:, 0:hidden_size] = Ui.T
        self.W[input_size:, hidden_size:2*hidden_size] = Uc.T
        self.W[input_size:, 2*hidden_size:3*hidden_size] = Uo.T

        # Bias
        self.b = np.concatenate((bi.T, bc.T, bo.T), axis=1)

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def forward_step(self, x_t, h_prev, c_prev):
        """
        Forward computation for a single time step
        Shapes are identical to the original implementation
        """
        x_t = x_t.reshape(1, -1)
        h_prev = h_prev.reshape(1, -1)
        c_prev = c_prev.reshape(1, -1)

        combined = np.concatenate((x_t, h_prev), axis=1)

        gates = np.dot(combined, self.W) + self.b

        i_raw = gates[:, 0:self.hidden_size]
        c_raw = gates[:, self.hidden_size:2*self.hidden_size]
        o_raw = gates[:, 2*self.hidden_size:3*self.hidden_size]

        i_t = self._sigmoid(i_raw)
        c_tilde = np.tanh(c_raw)
        o_t = self._sigmoid(o_raw)

        c_t = (1.0 - i_t) * c_prev + i_t * c_tilde
        h_t = o_t * np.tanh(c_t)

        return h_t.T, c_t.T

    def forward(self, X):
        """
        Forward pass for a full sequence
        Same input/output format as the original implementation
        """
        seq_length = X.shape[0]

        h_t = np.zeros((self.hidden_size, 1))
        c_t = np.zeros((self.hidden_size, 1))
        

        hidden_states = []

        for t in range(seq_length):
            x_t = X[t].reshape(-1, 1)
            h_t, c_t = self.forward_step(x_t, h_t, c_t)
            hidden_states.append(h_t.flatten())

        return np.array(hidden_states)


# In[5]:


import time
# ==========================================
# Testing the Optimized CIFG-LSTM Model
# ==========================================
if __name__ == "__main__":
    INPUT_SIZE = 10
    HIDDEN_SIZE = 16
    SEQ_LENGTH = 50

    np.random.seed(42)
    X_synthetic = np.random.randn(SEQ_LENGTH, INPUT_SIZE)

    model = CIFG_LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)

    all_hidden_states = model.forward(X_synthetic)

    print("=== Matrix Dimensions Verification ===")
    print(f"Input Time Series Shape: {X_synthetic.shape} -> (Seq_len=50, Input_size=10)")
    print(f"Optimized Weight Matrix Shape: {model.W.shape} -> (Input+Hidden, 3*Hidden)")
    print(f"Output Hidden States Shape: {all_hidden_states.shape} -> (Seq_len=50, Hidden_size=16)")
    assert all_hidden_states.shape == (SEQ_LENGTH, HIDDEN_SIZE), "Output dimension mismatch!"
    print("Dimensions verified successfully.\n")

    print("=== Extracting h_t at each time step ===")
    h_0 = np.zeros(HIDDEN_SIZE)
    print(f"Time Step 00 (Initial h_0) | Shape: {h_0.shape}")
    print(f"Values: {np.round(h_0, 3)}\n")

    for t in range(SEQ_LENGTH):
        h_t = all_hidden_states[t] 
        
        if t < 3:
            print(f"Time Step {t+1:02d} | Shape: {h_t.shape}")
            print(f"Values: {np.round(h_t, 3)}\n")
        elif t == 3:
            print("...\n[Steps 4 to 49 calculated and extracted, hidden for brevity]\n...")
        elif t == SEQ_LENGTH - 1:
            print(f"Time Step {t+1:02d} (Last Step) | Shape: {h_t.shape}")
            print(f"Values: {np.round(h_t, 3)}\n")
    # ==========================================
    # Performance Benchmarking
    # ==========================================
    print("=== Performance Benchmarking ===")
    num_runs = 1000  # تعداد تکرار برای محاسبه دقیق‌تر زمان
    
    # گرم کردن پردازنده (Warm-up)
    for _ in range(10):
        _ = model.forward(X_synthetic)
        
    # شروع اندازه‌گیری زمان
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        _ = model.forward(X_synthetic)
        
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_per_run = (total_time / num_runs) * 1000 # بر حسب میلی‌ثانیه
    
    print(f"Total time for {num_runs} forward passes: {total_time:.4f} seconds")
    print(f"Average time per forward pass: {avg_time_per_run:.4f} milliseconds")
    print("========================================\n")

