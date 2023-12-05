import matplotlib.pyplot as plt

def plot_model(steps, scores, avg_scores, epsilons, name):
    plt.cla();
    plt.clf();
    
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score', color=color)
    ax1.plot(steps, scores, label='Score', color=color)
    ax1.plot(steps, avg_scores, label='Average score', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([-2000, 0])

    color = 'tab:red'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(steps, epsilons, label='Epsilon', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Model learning curve')
    fig.tight_layout()
    fig.legend()

    plt.savefig(name)