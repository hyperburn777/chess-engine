import matplotlib.pyplot as plt

# Assuming train_losses and val_losses are already defined in your code
# For example:
# train_losses = [0.9, 0.7, 0.5, 0.4, 0.3]
# val_losses = [0.95, 0.8, 0.6, 0.55, 0.5]

def plot_loss_curves(train_losses, val_losses, filename="loss_plot.png"):
    plt.figure(figsize=(10, 6))
    
    # Plotting both lines
    plt.plot(train_losses, label='Training Loss', color='blue', lw=2)
    plt.plot(val_losses, label='Validation Loss', color='red', lw=2)
    
    # Formatting the chart
    plt.title('Training and Validation Loss Over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save and close
    plt.savefig(f"plots/{filename}")
    print(f"Plot saved successfully as {filename}")