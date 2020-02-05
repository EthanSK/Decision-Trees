from .data_read import data_read
from .data_set import Dataset, DataEntry
import matplotlib.pyplot as plt
import numpy as np

def compare_plots(data1: str, data2: str):
    data_set_1 = data_read(data1)
    data_set_2 = data_read(data2)
    
    index = np.arange(6)
    bar_width = 0.35

    count = { "A": 0, "C": 0, "E": 0, "G": 0, "O": 0, "Q": 0}
    for entry in data_set_1.entries:
        count[entry.label] += 1
    dictlist = []
    for key, value in count.items():
        dictlist.append(value)

    count2 = { "A": 0, "C": 0, "E": 0, "G": 0, "O": 0, "Q": 0}
    for entry in data_set_2.entries:
        count2[entry.label] += 1
    dictlist2 = []
    for key, value in count2.items():
        dictlist2.append(value)

    fig, ax = plt.subplots()
    dataset1bars = ax.bar(index, dictlist, bar_width,
        label="train_full.txt")
    dataset2bars = ax.bar(index+bar_width, dictlist2,
        bar_width, label="train_noisy.txt")
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    ax.set_title('Labels per data set')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(["A", "C", "E", "G", "O", "Q"])
    ax.legend()

    plt.show()
    plt.save()
    
    return





if __name__ == "__main__":
    compare_plots("data/train_full.txt", "data/train_noisy.txt")