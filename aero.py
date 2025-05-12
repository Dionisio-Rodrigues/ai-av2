import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('TkAgg')


class Adaline:

    def __init__(self, lr=0.01, random_state=1, epochs=50, tolerance=0.01):
        self.lr = lr
        self.random_state = random_state
        self.epochs = epochs
        self.W = None
        self.tolerance = tolerance

    def activation(self, u):
        return u

    def linear_combination(self, X):
        return X@self.W

    def loss_function(self, y, u, p):
        return np.sum((y - u)**2)/(2*p)
    
    def save_weights(self, filename="weights.npy"):
        np.savetxt(filename, self.W)

    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        p = X.shape[0]
        rgen = np.random.RandomState(self.random_state)
        self.W = rgen.normal(loc=0.0, size=X.shape[1], scale=0.01)
        mse = self.loss_function(y, self.linear_combination(X), p)

        for _ in range(self.epochs):

            if mse < self.tolerance:
                break

            for _x, _y in zip(X, y):
                u = _x @ self.W
                error = _y - u
                self.W += self.lr * error * _x
            
            mse = self.loss_function(y, self.linear_combination(X), p)
        
        self.save_weights()


    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        # self.W = np.load("weights.npy", allow_pickle=True)
        return self.linear_combination(X)


aero_data = np.loadtxt('./aerogerador.dat', 'float', delimiter='\t')

X = aero_data[:, 0]
Y = aero_data[:, 1]


fig = plt.figure()

ax_aero_data = fig.add_subplot(311)
ax_aero_data.scatter(X, Y)

X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

X_training = X[:int(Y.shape[0]*.70)]
X_test = X[int(Y.shape[0]*.70):]

y_training = Y[:int(Y.shape[0]*.70)]
y_test = Y[int(Y.shape[0]*.70):]

adaline = Adaline(epochs=1000, tolerance=0.01)
adaline.fit(X_training, y_training)

y_predicted = adaline.predict(X_test)


ax_aero_test = fig.add_subplot(412)
ax_aero_test.scatter(X_test, y_test)
ax_aero_test.set_xlim(X_test.min(), X_test.max())
ax_aero_test.set_ylim(y_test.min(), y_test.max())

ax_aero_predicted = fig.add_subplot(413)
ax_aero_predicted.scatter(X_test, y_predicted)
ax_aero_predicted.set_xlim(X_test.min(), X_test.max())
ax_aero_predicted.set_ylim(y_test.min(), y_test.max())

ax_aero_predicted = fig.add_subplot(414)
ax_aero_predicted.plot(X_test, y_predicted)
ax_aero_predicted.set_xlim(X_test.min(), X_test.max())
ax_aero_predicted.set_ylim(y_test.min(), y_test.max())

plt.tight_layout()
plt.show()
