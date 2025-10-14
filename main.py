from dataset import generate_dataset1, generate_dataset2
from gradient_descent import full_batch_gd, stochastic_gd, normal_equation, full_batch_gd_epoch,stochastic_gd_epoch

def main():
    datasets = {
        "Dataset 1": generate_dataset1(),
        "Dataset 2": generate_dataset2()
    }

    for name, (X, y) in datasets.items():
        print(f"\n--> Running {name} ...")

        theta_star = normal_equation(X, y)

        theta_batch, _, steps_batch = full_batch_gd(X, y, theta_star)
        theta_sgd, _, steps_sgd = stochastic_gd(X, y, theta_star)

        print(f"\n Results for {name}:")
        print(f"  True θ*:          {theta_star}")
        print(f"  Full-Batch θ:     {theta_batch}")
        print(f"  Stochastic θ:     {theta_sgd}")
        print(f"  Full-Batch Epochs: {steps_batch}")
        print(f"  Stochastic Epochs: {steps_sgd}\n")


        full_batch_gd_epoch(X, y, theta_star)
        stochastic_gd_epoch(X, y, theta_star)

if __name__ == "__main__":
    main()
