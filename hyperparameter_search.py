import itertools


learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
epoch_counts = [5, 10, 15, 20]
batch_sizes = [32, 64, 128]
dropout = [0.2, 0.4, 0.6, 0.8, 0.9]

hyperparams = [learning_rates, epoch_counts, batch_sizes, dropout]

permutations = list(itertools.product(*hyperparams))
print(len(permutations))

marcel = hyperparams[:105]
mahfuza = hyperparams[105:210]
nigel = hyperparams[210:315]
lambert = hyperparams[315:]

print(permutations)
