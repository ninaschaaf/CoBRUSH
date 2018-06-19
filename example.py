from hybrid import *
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# logging.basicConfig(filename='tictac_support_eval.log', level=logging.DEBUG)

# collect data
df = pd.read_csv('data/second_tictac.csv', header=0, sep=",")

# Mushroom
# Y = df['poisonous'].values
# X = df.drop('poisonous', axis=1)
# Yb = pd.read_csv('data/mlp_predictions.csv', sep=",", header=0).values[:, 0]  # Yb's shape has to be (X, )

# Tic-Tac-Toe
Y = df['xwins'].values
X = df.drop('xwins', axis=1)
Yb = pd.read_csv('data/mlp_predictions_tictac.csv', sep=",", header=0).values[:, 0]  # Yb's shape has to be (X, )

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)
Yb_train, Yb_test = train_test_split(Yb, test_size=0.3, shuffle=False)
print('X_train-shape: {}, X_test-shape: {}, Yb_train-shape: {}, Yb_test-shape: {}'.format(X_train.shape, X_test.shape, Yb_train.shape, Yb_test.shape))

model = hyb(X_train, y_train, Yb_train)
model.set_parameters(alpha=0.8, beta=100)
model.precompute()

model.generate_rulespace(5, 3, 500, method='forest')
maps, accuracy, covered = model.train(nIter=2000, print_message=False)
# test
pred, covered, yb = model.predict(X_test, y_test, Yb_test)
TP, FP, TN, FN = getConfusion(pred, y_test)
tpr = float(TP)/(TP+FN)
fpr = float(FP)/(FP+TN)
print('TP = {}, FP = {}, TN = {}, FN = {} \n accuracy = {}, tpr = {}, fpr = {}'
      .format(TP, FP, TN, FN, float(TP+TN)/(TP+TN+FP+FN), tpr, fpr))


# test_alpha = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 10, 100]
# test_beta = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 10, 100]
# for a in test_alpha:
#     for b in test_beta:
#         print('Alpha: {}, Beta: {}:'.format(a, b))
#         logging.info('Alpha: {}, Beta: {}:'.format(a, b))
#         model.set_parameters(alpha=a, beta=b)
#         model.precompute()
#
#         model.generate_rulespace(10, 3, 1000, method='forest')
#         maps, accuracy, covered = model.train(nIter=1000, print_message=False)
#         # test
#         pred, covered, yb = model.predict(X_test, y_test, Yb_test)
#         TP, FP, TN, FN = getConfusion(pred, y_test)
#         tpr = float(TP) / (TP + FN)
#         fpr = float(FP) / (FP + TN)
#         print('TP = {}, FP = {}, TN = {}, FN = {} \n accuracy = {}, tpr = {}, fpr = {}'
#               .format(TP, FP, TN, FN, float(TP + TN) / (TP + TN + FP + FN), tpr, fpr))
#         logging.info('TP = {}, FP = {}, TN = {}, FN = {} \n accuracy = {}, tpr = {}, fpr = {}'
#               .format(TP, FP, TN, FN, float(TP + TN) / (TP + TN + FP + FN), tpr, fpr))
