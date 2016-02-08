import numpy as np
from AbstractClassifier import AbstractClassfier


class MySvm(AbstractClassfier):
    def __init__(self):
        self._w = np.array([])
        self._mins = np.array([])  # Vector min value of each feature in training data. Serve us to normalize features
        self._maxs = np.array([])  # Vector max value of each feature in training data. Serve us to normalize features
		
		
        ############################################################################
        # We want to be able to control on the ratio of False Negative (when it is face, and SVM detect as non-face)
        # and False Positive (when it is non-face, and SVM detect as face).
        # for example, if we want that the SVM never err the kind of False Positive mistakes, we able to do it.
        # We do it by those parameters:
        ############################################################################
        self.false_positive_loss = 1 # when we want to *not err* False Positive, we increase the value of this parameter
        self.false_negative_loss = 1 # when we want to *not err* False Negative, we increase the value of this parameter
		self.error = 0  # Amount of errors with weidght
        self.simple_error = 0  # Amont of errors

		
    def _sub_gradient_loss(self, example, W, c, n):
		"""
        :param example: 
        :param W: w vector
        :param c: Conastant
        :param n: Number the examples in all our data
        :return: Derivative of loss function
        """
        (x, y) = example[:-1], example[-1]
        grad_loss = W / n
        if 1 - self._loss_value(y)*y * W.dot(x) > 0:
            grad_loss -= c*self._loss_value(y)*y * x
        return grad_loss

    def _loss_value(self, y):
		"""
		See documentation of __init__ function
		"""
        return self.false_negative_loss if y == 1 else self.false_positive_loss

    def _svm_c(self, examples, c, epoch):
        """
        This function learning the W vector
        :param examples: Matrix of examples, whith a target for each example (in last column)
        :param c: It's C of SVM [in section: C*max(1-w*x[i]*y[i])]
        :return: Vector of coefficients. when we multiply this vector with the vector of the object's features, we will get a positive result if the object is a face, otherwise, we will get a negative result.
        """
        num_ex = examples.shape[0]
        alpha = 1
        w_vec = np.zeros(examples.shape[1]-1)
        t = 0
        for _ in range(epoch * num_ex):
            r = examples[np.random.randint(num_ex)]
            t += 1
            w_vec -= alpha/t * self._sub_gradient_loss(r, w_vec, c, num_ex)
        return w_vec

    def learn(self, examples, c_arr=None, epoch=3, learning_part=0.7, cross_validation_times=5):
        """
        This function receive some possible valuse of C and chooses the best C (in additional, call to _svm_c that referred to above, and update the variables of the class
        :param examples: Vector of features vectors, and last column is target of the example
        :param c_arr: Array of C's candidates
        """
        if c_arr is None:
            c_arr = 2**np.arange(-5, 16, 2.0)

        examples = np.insert(examples, [-1], examples[:, :-1]**2, 1)
        n = examples.shape[0]  # n is amount of examples
        middle = round(n*learning_part) if learning_part > 0 else -1  # middle is present where to split examples to testing and learning. If middle <= 0, take all examples without last example
        # normalize
        x_mat = examples[:, :-1]
        mins = x_mat.min(axis=0)
        maxs = x_mat.max(axis=0) - mins
        maxs[maxs == 0] = 1  # If there are elements in maxs that they equals 0, exchange them with 1
        x_mat[:] = (x_mat - mins) / maxs
        c_len = c_arr.shape[0]  # c_len is a number of c's candidates
        errors = np.zeros(c_len)  # Vector of errors amount for each c's candidate
        # find best c:
		for j in range(c_len):
            c = c_arr[j]
            error = 0
            for _ in range(cross_validation_times):
                shuffled = np.random.permutation(examples)  # Mix examples
                learnings, testings = shuffled[:middle], shuffled[middle:]
                w_vec = self._svm_c(learnings, c, epoch)
                error = sum([(r[-1] * w_vec.dot(r[:-1]) < 0) * self._loss_value(r[-1]) for r in testings])
            if testings.shape[0] > 0:
                errors[j] += error/testings.shape[0]  # Change error[j] to percent 
        errors /= cross_validation_times
        result_c = c_arr[np.argmin(errors)]  # Choice best c
        w_vec = self._svm_c(examples, result_c, epoch)  # Learning w by the best c taht found
        #ending
        self._w = w_vec
        self._mins = mins  # Keep information required for normalize
        self._maxs = maxs  # Keep information required for normalize
        self.error = sum([(r[-1] * w_vec.dot(r[:-1]) < 0) * self._loss_value(r[-1]) for r in examples]) / n
        self.simple_error = sum([(r[-1] * w_vec.dot(r[:-1]) < 0) for r in examples]) / n

    def to_list(self):
		"""
        :return: List that keep in first row the values of W vector, required information for the normalization, sum errors (with weight) and amount error.
        """
        return [self._w, self._mins, self._maxs, self.error, self.simple_error]

    def from_list(self, list_):
		"""
        This function reading from list the required information for face detection: W vector, data of normalization, sum error (with weight) and amount error
        """
        if len(list_) != 5:
            raise ValueError('from_list: len(list_) has to be 5')
        self._w, self._mins, self._maxs, self.error, self.simple_error = list_

    def classify(self, x):
		"""
        This function classifying object as face or non-face
        :param x: Vector of features of the object
        :return: True if object is face, False otherwise
        """
        return 1 if self.valuefy(x) > 0 else -1

    def classify_vec(self, vec, axis=-1):
		"""
        The aim of this function is to activate the classify function over vector of features' vectors
        :param axis: Dimension to apply the classify
        :return: Boolean vector without the 'axis' dimension
        """
        return np.apply_along_axis(self.classify, axis, vec)

    def valuefy(self, x):
		 """
        The function multiplies W vector with X features' vector and return the value of result (not boolean, value).
        """
        return self._w.dot((np.hstack((x, x**2)) - self._mins) / self._maxs)

    def valuefy_vec(self, vec, axis=-1):
        return np.apply_along_axis(self.valuefy, axis, vec)

