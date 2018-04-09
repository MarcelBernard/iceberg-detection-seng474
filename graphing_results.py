import csv
import matplotlib.pyplot as plt
import numpy as np

with open('overfitting_results_hyperparams.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    validation_accuracy = reader.__next__()[1:]
    train_accuracy = reader.__next__()[1:]

validation_accuracy = [float(num) for num in validation_accuracy]
train_accuracy = [float(num) for num in train_accuracy]

plt.gca().set_color_cycle(['blue', 'orange'])

x = np.arange(40)
plt.plot(x, validation_accuracy, linewidth=2)
plt.plot(x, train_accuracy, linewidth=2)
plt.title('Training accuracy vs validation accuracy', fontsize=16)
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.legend(['validation accuracy', 'training accuracy'], loc='upper left')
plt.savefig('overfit.png')

plt.clf()

# Predict all not iceberg
dummy_score = 0.5305486284289277

# Average accuracy: 0.8303946687370601
# Average log loss: 0.5069125457153426
# Average precision: 0.8038455049069704
# Average recall: 0.8540000000000001
baseline_score = 0.8303946687370601

# Average accuracy: 0.8643995907072515
# Average log loss: 0.43207253899976805
# Average precision: 0.8294877285448315
# Average recall: 0.8960935953004334
extra_data_score = 0.8643995907072515

# Average accuracy: 0.8989963245569836
# Average log loss: 0.2714560386824273
# Average precision: 0.8860411618007781
# Average recall: 0.9020439594288355
hyperparam_score = 0.8989963245569836

categories = ('Baseline', 'Augmented\ndata', 'Hyperparameter\noptimization')
y_pos = [1.5, 3, 4.5]
scores = [baseline_score, extra_data_score, hyperparam_score]

plt.bar(y_pos, scores, align='center', alpha=0.5, color='blue')
plt.ylim(0.75, 1)
plt.xticks(y_pos, categories)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Classification Accuracies', fontsize=16)

plt.savefig('accuracies.png')