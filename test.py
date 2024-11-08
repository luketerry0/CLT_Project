import random
from main import main

# code to test many different parameters
choices = range(5)
batch_sizes = [100, 5000]
learning_rates_range = [0.01, 0.0001]
values_to_choose = 30
trials = 40
for i in range(values_to_choose):
    batch_size = random.randint(batch_sizes[0], batch_sizes[1])
    learning_rate = random.uniform(learning_rates_range[0], learning_rates_range[1])
    for trial in range(trials):
        main(batch_size, learning_rate)

# # use the same parameters for many runs in order to make a histogram
# transformations = (1, 2, 3)
# batch_size = 100
# learning_rate = 0.001
# trials = 1
# for i in range(trials):
#     main(transformations, batch_size, learning_rate)
