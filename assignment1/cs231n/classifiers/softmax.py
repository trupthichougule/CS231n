import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1] #10
    num_train = X.shape[0] #500. the minibatch number of images
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correctClass_score = np.exp(scores[y[i]])
        total_sum=0
        #denominator of softmax
        for j in xrange(num_classes):
            total_sum += np.exp(scores[j])  

        softmax = correctClass_score / total_sum
        loss += -np.log(softmax) 
        #find dW grad for incorrect and correct class 
        for j in xrange(num_classes):
            dW[:,j] += (np.exp(scores[j])/total_sum)*X[i]
        
        dW[:,y[i]] += -X[i]
          
    loss += reg*np.sum(W*W)
    loss = loss / num_train
    dW = dW / num_train
    dW += reg*W
    return loss, dW
    pass

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
  # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

    scores = X.dot(W)
    scores = np.exp(scores)
    scores_sums = np.sum(scores, axis=1) #for each X, sum along rows. (500,)
    correctClass_score = scores[range(num_train), y]
    loss = correctClass_score / scores_sums
    softmax = np.log(loss)
    loss = -np.sum(softmax)

    loss += reg*np.sum(W*W)  
    loss /= num_train  #average
    
    s = np.divide(scores, scores_sums.reshape(num_train, 1))
    s[range(num_train), y] = - (scores_sums - correctClass_score) / scores_sums
    dW = X.T.dot(s)
    dW /= num_train
    dW += reg*W
    pass

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

