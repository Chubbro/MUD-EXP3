import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

def line_plt(data):
    for ls,name in data:
        length = len(ls)
        x = np.linspace(start=0, stop=length, num=length, endpoint=False)
        plt.plot(x, ls, label=name)  # marker='*'
    plt.legend()
    plt.savefig(fname="loss.pdf",format="pdf")
    plt.show()


def fb_plt(data):
    mkr = ["o", "+", "s", ".", "x", "^", "v"]
    i = 0
    fig, ax = plt.subplots(figsize=(6, 5))

    for ls,name in data:
        length = ls.shape[1]
        x = np.linspace(start=0, stop=length, num=length, endpoint=False)
        std_ls = np.std(ls, axis=0)
        mean_ls = np.mean(ls, axis=0)
        usl = mean_ls + 2*std_ls
        lsl = mean_ls - 2*std_ls

        ax.fill_between(x, lsl, usl, alpha=.3, linewidth=0)
        ax.plot(x, mean_ls, label=name, marker=mkr[i], markevery=10000, markersize=4)  # marker='*'
        i += 1
        
    ax.legend()
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='x', useMathText=True)
    ax.ticklabel_format(style='sci', scilimits=(-1,3), axis='y', useMathText=True)
    ax.set_xlabel("t")
    ax.set_ylabel("cumulative loss")
    ax.set(title='max delay = 10')
    ax.legend(prop={'size': 10}, loc='upper left')
    fig.savefig(fname="loss.pdf", format="pdf")
    plt.tight_layout()
    plt.show()
