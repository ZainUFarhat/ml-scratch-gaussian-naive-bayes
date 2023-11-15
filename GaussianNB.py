# numpy
import numpy as np

# gaussian naive bayes
class GaussianNaiveBayes():

    """
    Description:
        My from scratch implementation of the Gaussian Naive Bayes Algorithm
    """

    # fit
    def fit(self, X, y):

        """
        Description:
            Fits our Gaussian Naive Bayes model
        
        Parameters:
            X: train features
            y: train labels
        
        Returns:
            None
        """

        # fetch number of samples and features of X
        N, num_features = X.shape
        # fetch classes
        self.classes = np.unique(y)
        # find number of classes
        num_classes = len(self.classes)

        # compute means, variances, and priors for each class
        self.mean = np.zeros((num_classes, num_features))
        self.var =  np.zeros((num_classes, num_features))
        self.priors = np.zeros(num_classes)

        # iterate through each unique class
        for (i, c) in enumerate(self.classes):

            # find all the features of 'c' class
            X_c = X[y == c]
            # compute the mean, variance, and priors for each class
            self.mean[i, :] = X_c.mean(axis = 0)
            self.var[i, :] = X_c.var(axis = 0)
            self.priors[i] = X_c.shape[0] / N
        
        # return
        return None
    
    # gaussian
    def gaussian(self, class_i, x):

        """
        Description:
            Probability density function of the Gaussian distribution

        Parameters:
            class_i: class of our feature row
            x: feature row
        
        Returns:
            p
        """

        # find the mean and variance of corresponding feature row
        mean, var = self.mean[class_i], self.var[class_i]

        # find numerator and denominator values for our gaussian pdf
        numerator = np.exp(-1 * ((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var) + 1e-9

        # compute probability
        p = numerator / denominator

        # return
        return p
    
    # bayes theorem
    def bayes_theorem(self, x):

        """
        Description:
            Finds the conditional probability by applying bayes theorem

        Parameters:
            x: single feature row of X

        Returns:
            label 
        """

        # empty list to hold our posteriors
        posteriors = []

        # compute posterior probability for each class
        for i in range(len(self.classes)):

            # prior
            prior = np.log(self.priors[i])
            # posterior
            posterior = np.sum(np.log(self.gaussian(i, x))) + prior
            posteriors.append(posterior)
        
        # find our label
        label = self.classes[np.argmax(posteriors)]

        # return
        return label

    # predict
    def predict(self, X):

        """
        Description:
            Predicts on our fitted Gaussian Naive Bayes Model
        
        Parameters:
            X: test features
        
        Returns:
            predictions
        """

        # predict by computing bayes theorem and finding corresponding label
        predictions = np.array([self.bayes_theorem(x) for x in X])

        # return
        return predictions