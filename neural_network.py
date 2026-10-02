import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size=8, learning_rate=0.1, epochs=10000, random_state=42):
        rng = np.random.default_rng(random_state)
        self.weights1 = rng.normal(0, np.sqrt(1 / input_size), (input_size, hidden_size))
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = rng.normal(0, np.sqrt(1 / hidden_size), (hidden_size, 1))
        self.bias2 = np.zeros((1, 1))
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.hidden_input = X @ self.weights1 + self.bias1
        self.hidden_output = np.tanh(self.hidden_input)
        self.output_input = self.hidden_output @ self.weights2 + self.bias2
        self.predictions = self.sigmoid(self.output_input)
        return self.predictions

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("X must be 2D and have the same number of rows as y.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Training data must contain only finite values.")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("This example supports binary labels: 0 or 1.")

        sample_count = X.shape[0]
        for epoch in range(self.epochs):
            predictions = self.forward(X)
            epsilon = 1e-9
            loss = -np.mean(
                y * np.log(predictions + epsilon)
                + (1 - y) * np.log(1 - predictions + epsilon)
            )
            self.loss_history.append(float(loss))

            output_delta = (predictions - y) / sample_count
            weights2_gradient = self.hidden_output.T @ output_delta
            bias2_gradient = np.sum(output_delta, axis=0, keepdims=True)

            hidden_delta = (output_delta @ self.weights2.T) * (
                1 - self.hidden_output ** 2
            )
            weights1_gradient = X.T @ hidden_delta
            bias1_gradient = np.sum(hidden_delta, axis=0, keepdims=True)

            self.weights2 -= self.learning_rate * weights2_gradient
            self.bias2 -= self.learning_rate * bias2_gradient
            self.weights1 -= self.learning_rate * weights1_gradient
            self.bias1 -= self.learning_rate * bias1_gradient

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.weights1.shape[0]:
            raise ValueError("X must be 2D with the same number of features used in training.")
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)
    y = np.array([0, 1, 1, 0])

    model = NeuralNetwork(
        input_size=2,
        hidden_size=8,
        learning_rate=0.1,
        epochs=10000,
        random_state=42,
    )
    model.fit(X, y)

    probabilities = model.predict_proba(X).flatten()
    predictions = model.predict(X).flatten()

    print("XOR demonstration")
    print("Inputs:       Predicted probability:    Predicted class:")
    for inputs, probability, prediction in zip(X, probabilities, predictions):
        print(f"{inputs}          {probability:.4f}                    {prediction}")
    print(f"Final loss: {model.loss_history[-1]:.6f}")
