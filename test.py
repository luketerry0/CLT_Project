from itertools import combinations
from main import main

choices = range(5)
batch_sizes = [1, 10, 25, 50, 100, 150, 200, 300]
learning_rates = [0.01, 0.001, 0.0005, 0.0001]
for i in [1,2,3,4]:
    transform_indices = list(combinations(choices, i))
    for indices in transform_indices:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                main(indices, batch_size, learning_rate)