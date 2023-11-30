import matplotlib.pyplot as plt
import csv

def read_histogram_data(filename):
    bins = []
    counts = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            bins.append(int(row[0]))
            counts.append(int(row[1]))
    return bins, counts

def plot_histogram(bins, counts):
    plt.bar(bins, counts, width=1.0, edgecolor='black')
    plt.xlabel('Bin Number')
    plt.ylabel('Count')
    plt.title('Histogram')
    plt.savefig('histogram.png')

def main():
    filename = 'histogram.csv'  # replace with your file name
    bins, counts = read_histogram_data(filename)
    plot_histogram(bins, counts)

if __name__ == "__main__":
    main()
