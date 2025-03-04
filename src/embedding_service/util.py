import matplotlib.pyplot as plt


def plot_losses(epoch_losses: list[float]):
    # Plot the losses
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_losses, marker="o", linestyle="-", label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig("training_outputs/loss.png")
    plt.close()
