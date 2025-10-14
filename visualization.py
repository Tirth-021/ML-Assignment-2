import numpy as np
import matplotlib.pyplot as plt

def plot_loss_curve(losses, title="Loss vs Epochs", show=True):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses)+1), losses, label="Loss", linewidth=3)
    plt.xlabel("Epochs")   
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(range(1, len(losses)+1))
    if show:
        plt.show()


def plot_contour(X, y, theta_history, theta_star, title="Contour Plot"):
    theta0_vals = np.linspace(theta_star[0] - 10, theta_star[0] + 10, 100)
    theta1_vals = np.linspace(theta_star[1] - 10, theta_star[1] + 10, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    for i, t0 in enumerate(theta0_vals):
        for j, t1 in enumerate(theta1_vals):
            t = np.array([t0, t1])
            J_vals[i, j] = (1/(2*len(y))) * np.sum((X.dot(t) - y)**2)

    J_vals = J_vals.T
    """
    plt.contour(theta0_vals, theta1_vals, J_vals)
    Matplotlib expects that:

    The first axis (rows) corresponds to the y-axis variable (θ₁)

    The second axis (columns) corresponds to the x-axis variable (θ₀)

    But in your array:

    First axis (rows) = θ₀

    Second axis (columns) = θ₁
    SO we are transposing J_vals
    """

    plt.figure(figsize=(6, 5))
    plt.contour(theta0_vals, theta1_vals, J_vals,  cmap='jet')
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.title(title)
    plt.plot(theta_star[0], theta_star[1], 'rx', markersize=10, label='True θ*')
    plt.plot([t[0] for t in theta_history], [t[1] for t in theta_history],
             'o-', color='orange', label='Path')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
