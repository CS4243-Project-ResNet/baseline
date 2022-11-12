import sys
import matplotlib.pyplot as plt


def plot_graph(data, filename):
    train_losses, train_accs, val_losses, val_accs = data
    x_axis = range(len(train_losses))
    plt.plot(x_axis, train_losses, label="Train Loss")
    plt.plot(x_axis, val_losses, label="Val Loss")
    plt.plot(x_axis, train_accs, label="Train Acc")
    plt.plot(x_axis, val_accs, label="Val Acc")
    plt.legend()
    plt.savefig(f'{filename}.png')

def parse(filename):
    data = []

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    with open(f'{filename}.out', 'r') as f:
        lines = f.readlines()

        for i in range(0, len(lines), 6):
            if (len(lines) - i < 6): 
                break

            epoch = int(lines[i].split()[1].split('/')[0])
            line2_split = lines[i + 2].split()
            train_loss = float(line2_split[2])
            train_losses.append(train_loss)
            train_acc = float(line2_split[4])
            train_accs.append(train_acc)
            line3_split = lines[i + 3].split()
            val_loss = float(line3_split[2])
            val_losses.append(val_loss)
            val_acc = float(line3_split[4])
            val_accs.append(val_acc)
   
    return train_losses, train_accs, val_losses, val_accs

if __name__ == "__main__":
    filename = sys.argv[1]
    data = parse(filename)
    plot_graph(data, filename)