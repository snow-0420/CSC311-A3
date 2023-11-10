'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for digit in np.arange(0, 10):
        digits_label = data.get_digits_by_label(train_data, train_labels, digit)
        means[digit] = np.sum(digits_label, axis=0) / digits_label.shape[0]
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    mean = compute_mean_mles(train_data, train_labels)
    for digit in np.arange(0, 10):
        digits_label = data.get_digits_by_label(train_data, train_labels, digit)
        covariances[digit] = np.matmul((digits_label - mean[digit]).T, (digits_label - mean[digit])) / digits_label.shape[0]
        covariances[digit] += np.identity(64) * 0.01
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    likelihood = np.zeros((digits.shape[0], 10))
    for digit in np.arange(0, 10):
        term1 = ((2 * np.pi) ** (-digits.shape[1] / 2)) * (np.linalg.det(covariances[digit]) ** (-1 / 2))
        term2 = np.exp((-1 / 2) * np.diag((digits - means[digit]) @ np.linalg.inv(covariances[digit]) @ (digits - means[digit]).T))
        likelihood[:, digit] = np.log(term1 * term2)
    return likelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    numerator = generative_likelihood(digits, means, covariances) + np.log(1 / 10)
    denominator = np.logaddexp.accumulate(numerator, axis=1)[:, -1:]
    return numerator - denominator


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    sum_likelihood = 0
    for x_i in np.arange(0, digits.shape[0]):
        sum_likelihood += cond_likelihood[x_i, int(labels[x_i])]
    return sum_likelihood / digits.shape[0]

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    # part (a)
    print("Average conditional log likelihood for training: {0}".format(avg_conditional_likelihood(train_data, train_labels, means, covariances)))
    print("Average conditional log likelihood for test: {0}".format(avg_conditional_likelihood(test_data, test_labels, means, covariances)))

    # part (b)
    pred_train = classify_data(train_data, means, covariances)
    accuracy_train = np.sum(pred_train == train_labels) / train_data.shape[0]
    print("Accuracy on training: {0}".format(accuracy_train))
    pred_test = classify_data(test_data, means, covariances)
    accuracy_test = np.sum(pred_test == test_labels) / test_data.shape[0]
    print("Accuracy on test: {0}".format(accuracy_test))

    # part (c)
    covariances_diag = np.zeros_like(covariances)
    for digit in np.arange(0, 10):
        covariances_diag[digit] = np.diag(np.diag(covariances[digit]))
    print("Average conditional log likelihood for training with diagonal covariance: {0}".format(avg_conditional_likelihood(train_data, train_labels, means, covariances_diag)))
    print("Average conditional log likelihood for test with diagonal covariance: {0}".format(avg_conditional_likelihood(test_data, test_labels, means, covariances_diag)))
    pred_train = classify_data(train_data, means, covariances_diag)
    accuracy_train = np.sum(pred_train == train_labels) / train_data.shape[0]
    print("Accuracy on training with diagonal covariance: {0}".format(accuracy_train))
    pred_test = classify_data(test_data, means, covariances_diag)
    accuracy_test = np.sum(pred_test == test_labels) / test_data.shape[0]
    print("Accuracy on test with diagonal covariance: {0}".format(accuracy_test))

if __name__ == '__main__':
    main()
