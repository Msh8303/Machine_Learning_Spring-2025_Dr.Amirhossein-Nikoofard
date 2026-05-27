#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# # Generating Data

# In[2]:


# Generate the "Habitable Zone" dataset
X, y = make_circles(n_samples=600, noise=0.05, factor=0.5, random_state
=42)
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2,
random_state=42)


# # Plotting Dataset

# In[3]:


# Plotting the dataset
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', label='Uninhabitable (Class 0)')
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', label='Habitable (Class 1)')
plt.xlabel('Distance from Star (X1)')
plt.ylabel('Atmospheric Thickness (X2)')
plt.legend()
plt.title('Exoplanet Habitable Zone')
plt.show()


# # Generating logisticRegression Model

# In[4]:


# Generate model logisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


# prediction on model
y_pred = model.predict(X_test)


# # Evaluating on model

# In[11]:


# evaluation on model
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy on the test set: {accuracy * 100:.2f}%\n")
# classification Report(Precision, Recall, F1-Score)
print(classification_report(y_test, y_pred))


# # Generating meshgrid

# In[7]:


x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


# # Plotting decision boundary for logisticRegression

# In[8]:


plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr') 
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', edgecolor='k', label='Uninhabitable (Class 0)')
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', edgecolor='k', label='Habitable (Class 1)')

plt.xlabel('Distance from Star (X1)')
plt.ylabel('Atmospheric Thickness (X2)')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()


# In[9]:


from sklearn.neural_network import MLPClassifier


# # Generating MLP model with one Hidden layer

# In[19]:


# 2. تعریف مدل MLP با "تنها یک لایه پنهان" شامل 4 نورون
# activation='relu' : تابع فعال‌سازی لایه پنهان
# solver='lbfgs' : بهینه‌سازی که برای داده‌های کم و شبکه‌های کوچک بسیار عالی و همگرا عمل می‌کند
mlp_model = MLPClassifier(hidden_layer_sizes=(4,), 
                          activation='relu', 
                          solver='lbfgs', 
                          max_iter=1000, 
                          random_state=42)


# # Training Model

# In[20]:


mlp_model.fit(X_train, y_train)


# # Prediction on Test set

# In[21]:


y_pred = mlp_model.predict(X_test)


# # Evaluation on test set

# In[22]:


accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy on the test set: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# # Training Mlp model with adam optimization

# In[26]:


mlp_model = MLPClassifier(hidden_layer_sizes=(4,), 
                          activation='relu', 
                          solver='adam', 
                          max_iter=1000, 
                          random_state=42)


# # Training Model

# In[27]:


mlp_model.fit(X_train, y_train)


# # Prediction on Test set

# In[28]:


y_pred = mlp_model.predict(X_test)


# # Evaluation on test set

# In[29]:


accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy on the test set: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# # Plotting Decision Boundary for neurons

# In[24]:


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
neurons = [1, 2, 3, 4]

for i, ax in enumerate(axes.ravel()):
    n = neurons[i]
    
    # تعریف و آموزش مدل با n نورون در لایه پنهان
    mlp = MLPClassifier(hidden_layer_sizes=(n,), 
                        activation='relu', 
                        solver='lbfgs', 
                        max_iter=2000, 
                        random_state=42)
    mlp.fit(X_train, y_train)
    
    # پیش‌بینی برای تمام نقاط شبکه (Meshgrid)
    Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # محاسبه دقت روی داده‌های تست
    acc = mlp.score(X_test, y_test)
    
    # رسم مرز تصمیم
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    ax.scatter(X[y==0, 0], X[y==0, 1], color='red', edgecolor='k', s=20)
    ax.scatter(X[y==1, 0], X[y==1, 1], color='blue', edgecolor='k', s=20)
    
    ax.set_title(f'{n} Neuron(s) | Test Accuracy: {acc*100:.1f}%')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

plt.tight_layout()
plt.show()


# # Training models with sigmoid or Relu

# In[25]:


mlp_sigmoid = MLPClassifier(hidden_layer_sizes=(10, 10, 10), 
                            activation='logistic', 
                            solver='sgd', 
                            learning_rate_init=0.01, 
                            max_iter=1000, 
                            random_state=42)


mlp_relu = MLPClassifier(hidden_layer_sizes=(10, 10, 10), 
                         activation='relu', 
                         solver='sgd', 
                         learning_rate_init=0.01, 
                         max_iter=1000, 
                         random_state=42)


print("در حال آموزش شبکه با Sigmoid...")
mlp_sigmoid.fit(X_train, y_train)

print("در حال آموزش شبکه با ReLU...")
mlp_relu.fit(X_train, y_train)


plt.figure(figsize=(10, 6))
plt.plot(mlp_sigmoid.loss_curve_, label='Sigmoid Loss', color='red', linewidth=2)
plt.plot(mlp_relu.loss_curve_, label='ReLU Loss', color='blue', linewidth=2)

plt.title('Loss Curve Comparison: Sigmoid vs ReLU (3 Hidden Layers)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


print(f"Test Accuracy (Sigmoid): {mlp_sigmoid.score(X_test, y_test)*100:.2f}%")
print(f"Test Accuracy (ReLU): {mlp_relu.score(X_test, y_test)*100:.2f}%")

