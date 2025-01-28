import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from Network import NeuralNetwork

# Load MNIST data
def load_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32)
    Y = mnist.target.astype(np.int32)

    # Normalize pixel values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert labels to one-hot encoding
    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(Y.reshape(-1, 1))

    return X.T, Y.T

def preprocess_data():
    X, Y = load_mnist()

    # Select only 500 samples for faster training and testing
    X, _, Y, _ = train_test_split(X.T, Y.T, train_size=500, random_state=42)

    # Split into training and testing subsets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train.T, X_test.T, Y_train.T, Y_test.T

if __name__ == "__main__":
    np.random.seed(1)

    # Preprocess data
    X_train, X_test, Y_train, Y_test = preprocess_data()

    # Define the network architecture
    layer_dims = [784, 64, 32, 10]  # 4 layer network for testing
    nn = NeuralNetwork(layer_dims, learning_rate=0.01)

    # Train the neural network
    print("Training the neural network on a subset of MNIST...")
    costs = nn.train(X_train, Y_train, num_iterations=1000)

    # Test the model
    predictions = nn.predict(X_test)
    accuracy = np.mean(np.argmax(predictions, axis=0) == np.argmax(Y_test, axis=0))
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
