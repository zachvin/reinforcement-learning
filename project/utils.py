import matplotlib.pyplot as plt

def plot_model(steps, scores, epsilons, name):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Score', color=color)
    ax1.plot(steps, scores, label='Score', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([-2000, 0])

    color = 'tab:red'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(steps, epsilons, label='Epsilon', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Model learning curve')
    fig.tight_layout()

    plt.savefig(name)

def plot_loss(steps, loss, name):
    plt.clf()
    plt.cla()

    plt.plot(steps, loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('NN Loss')

    name = 'plots/loss.png'

    plt.savefig(name)
