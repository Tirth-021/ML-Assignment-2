import numpy as np
import matplotlib.pyplot as plt



# STEP 1: DATA GENERATION

np.random.seed(42)
num_samples = 100

# Generate feature and target with noise
x1 = np.random.uniform(0, 1000, num_samples)
f_x = 3 * x1 + 2
eps = np.random.randn(num_samples)
y = f_x + eps

# Feature matrix and target vector
X = np.c_[np.ones(num_samples), x1]   # shape (100, 2)
y = y.reshape(-1, 1)                  # shape (100, 1)


# STEP 2: theta*

theta_star = np.linalg.inv(X.T @ X) @ X.T @ y  


# STEP 3: FULL-BATCH GRADIENT DESCENT FUNCTION

def gradient_descent(X, y, theta_star, lr, eps=1e-3, max_iter=200000, seed=None):
    # Optional seed control (None = random every time)
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(None)

    # Random initial theta
    theta = np.random.randn(X.shape[1], 1)
    mse_history = []

    for i in range(max_iter):
        y_pred = X @ theta
        grad = (2 / len(y)) * (X.T @ (y_pred - y))
        theta -= lr * grad
        mse = np.mean((y - y_pred) ** 2)
        mse_history.append(mse)

        # Convergence condition
        if np.linalg.norm(theta - theta_star) < eps:
            return theta, i + 1, mse_history
        
    print(f"In given {max_iter} we didn't reach convergence criteria")

    return theta, max_iter, mse_history



# STEP 4: RUN WITHOUT SCALING

lr_unscaled = 0.0000001  # very small learning rate 
theta_unscaled, iter_unscaled, mse_unscaled = gradient_descent(
    X, y, theta_star, lr_unscaled
)


# STEP 5: APPLY Z-SCORE NORMALIZATION

x_scaled = (x1 - np.mean(x1)) / np.std(x1)
X_scaled = np.c_[np.ones(num_samples), x_scaled]
theta_star_scaled = np.linalg.inv(X_scaled.T @ X_scaled) @ X_scaled.T @ y

# After scaling, we can use a much larger learning rate
lr_scaled = 0.1
theta_scaled, iter_scaled, mse_scaled = gradient_descent(
    X_scaled, y, theta_star_scaled, lr_scaled
)


# STEP 6: RESULTS

print(f"Iterations required (Unscaled): {iter_unscaled}")
print(f"Iterations required (Scaled): {iter_scaled}")
print("\nTheta* (Unscaled):\n", theta_star)
print("\nTheta (Unscaled):\n", theta_unscaled)
print("\nTheta* (Scaled):\n", theta_star_scaled)
print("\nTheta (Scaled):\n", theta_scaled)


# STEP 7: PLOT MSE vs ITERATIONS

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(mse_unscaled, color='blue')
plt.title("MSE vs Iterations (Unscaled Feature)")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")

plt.subplot(1, 2, 2)
plt.plot(mse_scaled, color='orange')
plt.title("MSE vs Iterations (Z-score Scaled Feature)")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")

plt.tight_layout()
plt.show()
