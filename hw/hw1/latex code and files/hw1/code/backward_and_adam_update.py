import numpy as np
def backward_and_adam_update(X ,A1, W2, dZ2, W1,b1,m,t,m_t,v_t):
    """_summary_

    Args:
    
        X (n_x, m): input matrix
        A1 (n_h, m): the output of the first hidden layer passed from ReLU
        W2 (n_y, n_h): weights of the second layer
        dZ2 (n_y, m): The gradient of the loss function with respect to z2 
        W1 (n_h, n_x): weights of the first layer
        b1 (n_h, 1): the bias of the first layer
        m (scaler): the size of mini-batch
        t : current timestamp for adam
        m_t (n_h, n_x): 1st moment for W1
        v_t (n_h, n_x): 2nd moment 
    """
    
    # dZ1 - ReLU derivative
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * (A1 > 0)
    
    #dW1 , db1
    dW1 = (1 / m) * (dZ1 @ X.T)
    db1 = (1 / m) * np.sum(dZ1, axis = 1, keepdims = True)
    
    
    #adam parameters
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.1
    epsilon = 1e-8
    
    #1st moment and 2nd moment updated
    m_t_new = beta1 * m_t + (1 - beta1) * dW1
    v_t_new = beta2 * v_t + (1 - beta2 ) * (dW1 * dW1)
    
    m_hat = m_t_new / (1 - beta1 ** t)
    v_hat = v_t_new / (1 - beta2 ** t) 
    
    w1_updated = W1 - lr * (m_hat / (np.sqrt(v_hat) + epsilon))    
    
    return w1_updated, m_t_new, v_t_new
    
