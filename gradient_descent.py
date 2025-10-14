import numpy as np
from visualization import plot_loss_curve, plot_contour

# Helper Functions

def gradient(X, y, theta):
    m = len(y)
    return (1/m) * X.T.dot(X.dot(theta) - y)

def compute_loss(X, y, theta):
    m = len(y)
    return (1/(2*m)) * np.sum((X.dot(theta) - y)**2)

def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Gradient Descent Methods


def full_batch_gd(X, y, theta_star, lr=0.01, epsilon=0.001, max_epochs=15, theta_init=None):  
    theta = np.zeros(X.shape[1]) if theta_init is None else theta_init.copy()  
    losses = []
    theta_history = [theta.copy()]

    for epoch in range(1, max_epochs + 1):   
        grad = gradient(X, y, theta)
        theta -= lr * grad
        loss = compute_loss(X, y, theta)
        losses.append(loss)
        theta_history.append(theta.copy())

        if np.linalg.norm(theta - theta_star) < epsilon:
            print(f"Full-batch GD converged in {epoch} epochs.")
            break
 
    plot_loss_curve(losses, f"Full-Batch GD: Loss vs Epochs", show=True)
    plot_contour(X, y, theta_history, theta_star, "Full-Batch Gradient Descent Path")
    return theta, losses, epoch


def stochastic_gd(X, y, theta_star, lr=0.01, epsilon=0.001, max_epochs=15, theta_init=None):  
    theta = np.zeros(X.shape[1]) if theta_init is None else theta_init.copy() 
    losses = []
    theta_history = [theta.copy()]
    m = len(y)

    for epoch in range(1, max_epochs + 1):   
        epoch_losses = []
        for i in range(m):
            xi = X[i, :].reshape(1, -1)
            yi = y[i]
            grad = xi.T.dot(xi.dot(theta) - yi)
            theta -= lr * grad
            epoch_losses.append(compute_loss(X, y, theta))

        losses.append(np.mean(epoch_losses))   # Log mean loss per epoch
        theta_history.append(theta.copy())

        if np.linalg.norm(theta - theta_star) < epsilon:
            print(f"Stochastic GD converged in {epoch} epochs.")
            break

    plot_loss_curve(losses, f"Stochastic GD: Loss vs Epochs", show=True)
    plot_contour(X, y, theta_history, theta_star, "Stochastic Gradient Descent Path")
    return theta, losses, epoch



def full_batch_gd_epoch(X, y, theta_star, lr=0.01, epsilon=0.001): 
    steps =  []
    for i in range (20): 
        np.random.seed()
        theta = np.random.randn(X.shape[1]) 
        epoch = 0 
        
        while np.linalg.norm(theta - theta_star) >= epsilon: 
            grad = gradient(X, y, theta) 
            theta -= lr * grad 
            epoch += 1 
        steps.append(epoch)

    avg_epochs = np.mean(steps)
    std_epochs = np.std(steps)
    
    print(f"Full-batch GD converged in {avg_epochs:.2f} ± {std_epochs:.2f} epochs on average.")
    
    return avg_epochs, std_epochs

def stochastic_gd_epoch(X, y, theta_star, lr=0.005, epsilon=0.001):
    steps = []  
    c = 0
    m = len(y)

    for i in range(20):
        np.random.seed()  
        theta = np.random.randn(X.shape[1])
        epoch = 0

        # loop until convergence
        while np.linalg.norm(theta - theta_star) >= epsilon:
            # shuffle dataset each epoch for SGD randomness
            indices = np.random.permutation(m)
            for idx in indices:
                xi = X[idx, :].reshape(1, -1)
                yi = y[idx]
                grad = xi.T.dot(xi.dot(theta) - yi)
                theta -= lr * grad
            grad *= 0.99
            epoch += 1  # one full pass = one epoch
        c += epoch
        steps.append(epoch)

        if(c > 50000): 
            print("Sorry")
            break

    avg_epochs = np.mean(steps)
    std_epochs = np.std(steps)

    print(f"Stochastic GD converged in {avg_epochs:.2f} ± {std_epochs:.2f} epochs on average.")

    return avg_epochs, std_epochs
