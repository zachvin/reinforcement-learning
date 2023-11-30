import matplotlib.pyplot as plt

def plot_model(steps, scores, epsilons, name):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Score', color=color)
    ax1.plot(steps, scores, label='Score', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:red'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(steps, epsilons, label='Epsilon', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Model learning curve')
    fig.tight_layout()

    plt.savefig(name)
