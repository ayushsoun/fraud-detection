import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import sklearn


print('python : {}'.format(sys.version))
print('Numpy: {}'.format(np.__version__))
print('pandas: {}'.format(pd.__version__))

print('seaborn: {}'.format(sns.__version__))
print('scipy: {}'.format(scipy.__version__))
#import dataset file
credit_card_data = pd.read_csv('creditcard.csv')
print(credit_card_data.columns)
print(credit_card_data.shape)
print(credit_card_data.describe())
credit_card_data.hist(figsize =(20,20))
plt.show()
data = credit_card_data.sample(frac=0.1, random_state = 1)
#dtetemine number of fraud transaction in dataset
Fraud = data[data['Class']==1]
valid = data[data['Class'] == 0]
outlier_fraction = len(Fraud)/float(len(valid))
print(outlier_fraction)
print('Fraud Cases: ',format(len(data[data['Class']==1])))
print('valid transactions: {}'.format(len(data[data['Class'] == 0])))

#correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat , vmax=.8 , square = True)
plt.show()

# get all the columns from datagrame
columns = data.columns.tolist()

# filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]
target = "Class"

X = data[columns]
Y = data[target]

#print shapes
print(X.shape)
print(Y.shape)

#print(credit_card_data)

# 1. shuffled/randmize data
# 2. one_hot Encoding (function used for data 0 and 1)
# 3 . normalize
# 4 . splitting up X/Y values
# 5 . convert to numpy array
# 6 . splitting our final data X/Y test/train

shuffled_data = credit_card_data.sample(frac=1)
one_hot_data = pd.get_dummies(shuffled_data, columns=['Class'])
normalized_data = (one_hot_data - one_hot_data.min()) / (one_hot_data.max() - one_hot_data.min())
df_x = normalized_data.drop(['Class_0','Class_1'], axis =1)
df_y = normalized_data[['Class_0','Class_1']]
ar_x ,ar_y = np.asarray(df_x.values, dtype='float32'), np.asarray(df_y.values, dtype='float32')
train_size = int(0.8 * len(ar_x))
(row_x_train, row_y_train) = (ar_x [:train_size], ar_y[:train_size])
(row_x_test, row_y_test) = (ar_x [train_size:], ar_y[train_size:])


count_legist , count_fraud = np.unique(credit_card_data['Class'],return_counts = True)[1]
fraud_ratio = float(count_fraud / (count_legist + count_fraud))
print("percentage of fraudulent transaction: ",fraud_ratio)
print("total number of fraud transactions: ", count_fraud)
weighting = 1/ fraud_ratio
row_y_train[:, 1] = row_y_train[:, 1] * weighting
# building computational graph


#tensorflow
input_dimension = ar_x.shape[1]
output_dimension = ar_y.shape[1]
nlc1 = 100
nlc2 = 150
x_train_model = tf.placeholder(tf.float32, [None, input_dimension], name='x_train')
y_train_model = tf.placeholder(tf.float32,[None, output_dimension], name='y_train')

x_test_model = tf.constant(row_x_test, name = 'x_test')
y_test_model = tf.constant(row_y_test, name = 'y_test')

weight_1_model = tf.Variable(tf.zeros([input_dimension, nlc1]) , name = 'weight_1')
biases_1_model = tf.Variable(tf.zeros([nlc1]), name = 'biases_1')

weight_2_model = tf.Variable(tf.zeros([nlc1, nlc2]) , name = 'weight_2')
biases_2_model = tf.Variable(tf.zeros([nlc2]), name = 'biases_2')
weight_3_model = tf.Variable(tf.zeros([nlc2, output_dimension]) , name = 'weight_3')
biases_3_model = tf.Variable(tf.zeros([output_dimension]), name = 'biases_3')

def network(input_tensor):
    layer1 =tf.nn.sigmoid(tf.matmul(input_tensor, weight_1_model) + biases_1_model)
    layer2 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(layer1, weight_2_model) + biases_2_model), 0.85)
    layer3 = tf.nn.softmax(tf.matmul(layer2 , weight_3_model) + biases_3_model)
    return layer3

y_train_prediction = network(x_train_model)
y_test_prediction = network(x_test_model)

cross_entropy = tf.losses.softmax_cross_entropy(y_train_model, y_train_prediction)
optimizer = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

def calculate_accuracy(actual, predicted):
    actual= np.argmax(actual,1)
    predicted = np.argmax(predicted,1)
    return (100 * np.sum(np.equal(predicted, actual)) / predicted.shape[0])

num_epochs = 100
import time
with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(num_epochs):
        start_time = time.time()
        _, cross_entropy_score = session.run([optimizer, cross_entropy],
                                             feed_dict={x_train_model: row_x_train, y_train_model:row_y_train})
        if epoch % 10 == 0:
            timer = time.time() - start_time
            print('epoch:{}'.format(epoch),'current loss:{0:.4f}'.format(cross_entropy_score),'Elapsed time:{0:.2f} seconds'.format(timer))
            final_y_test = y_test_model.eval()
            final_y_test_prediction = y_test_prediction.eval()
            final_accuracy =  calculate_accuracy(final_y_test,final_y_test_prediction)
            print('Currrent Accuracy:{0:.2f}%'.format(final_accuracy))
    final_y_test = y_test_model.eval()
    final_y_test_prediction = y_test_prediction.eval()
    final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction)
    print('final Accuracy:{0:.2f}%'.format(final_accuracy))
final_fraud_y_test = final_y_test[final_y_test[:, 1] == 1]
final_fraud_y_test_prediction = final_y_test_prediction[final_y_test[:, 1] == 1]
final_fraud_test_accuracy = calculate_accuracy(final_fraud_y_test, final_fraud_y_test_prediction)
print('final fraud specific accuracy: {0:.2f}%'.format(final_fraud_test_accuracy))

