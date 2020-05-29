import matplotlib
import matplotlib.pyplot as plt
import numpy as np



def bar_chart(Y, y_label, width=0.35):
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    labels = [r"$f_1$", r"$f_2$", r"$f_3$", r"$f_4$", r"$f_5$"]
    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots()
    y = np.log(Y)
    rects1 = ax.bar(x - width/2, y[0], width, label='Men')
    rects2 = ax.bar(x + width/2, y[1], width, label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()





if __name__ == "__main__":
    values = np.array([[38, 11208, 140, 1041, 24],
                       [2, 48, 14, 58, 3]])
    bar_chart(values, "dist")
