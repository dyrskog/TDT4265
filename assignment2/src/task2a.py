import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    mean = np.mean(X)
    std = np.std(X)
    print(f'mean = {mean}, std = {std}')
    X = (X.astype(float)-mean)/std

    # Bias trick
    bias = np.ones((X.shape[0], 1))
    X = np.append(X, bias, axis=1)
    
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    
    Cn = targets * np.log(outputs)
    return -Cn.sum(axis=1).mean()


def sigmoid(z: np.ndarray):
    """
    Args:
        z: input vector
    Returns:
        Sigmoid function of z (np.ndarray)
    """
    return 1/(1 + np.exp(-z))


def improved_sigmoid(z: np.ndarray):
    return 1.7159*np.tanh(z*(2/3))


def sigmoid_derivative(z: np.ndarray):
    sig = sigmoid(z)
    return sig*(1-sig)


def improved_sigmoid_derivative(z: np.ndarray):
    return (1 - np.tanh(z*(2/3))**2)*1.7159*(2/3)


def softmax(z: np.ndarray):
    exp_sum = np.exp(z).sum(axis=1, keepdims=True)
    return np.exp(z)/exp_sum


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3b hyperparameter
                 use_improved_weight_init: bool,  # Task 3a hyperparameter
                 use_relu: bool  # Task 4 hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)

            if self.use_improved_weight_init:
                # Init with weights within a normal distribution
                mean = 0
                sd = 1/np.sqrt(size)
                w = np.random.normal(mean, sd, size=w_shape)
            else:
                # Init with random weights in range [-1 1]
                w = np.random.uniform(-1, 1, w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...
        self.node_inputs = []
        self.activation_inputs = []

        X = X
        for layer in range(len(self.ws)):
            self.node_inputs.append(X)
            X = X.dot(self.ws[layer])
            if layer == (len(self.ws) - 1):
                X = softmax(X)
            else:
                self.activation_inputs.append(X)
                if self.use_improved_sigmoid:
                    X = improved_sigmoid(X)
                else:
                    X = sigmoid(X)
        return X

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        self.zero_grad()
        self.grads = []
        batch_size = X.shape[0]
        # Last layer
        delta = -(targets - outputs)
        
        # Hidden layers 
        for layer in range(len(self.ws)-1, 0, -1):
            grad = (delta.T.dot(self.node_inputs[layer]))/batch_size
            self.grads = [grad.T] + self.grads

            if self.use_improved_sigmoid:
                delta = improved_sigmoid_derivative(self.activation_inputs[layer-1])*delta.dot(self.ws[layer].T)
            else:
                delta = sigmoid_derivative(self.activation_inputs[layer-1])*delta.dot(self.ws[layer].T)    

        # Avoid using sigmoid function on input layer
        grad = (delta.T.dot(self.node_inputs[0]))/batch_size
        self.grads = [grad.T] + self.grads


        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    y_ohe = np.zeros((Y.shape[0],num_classes))
    y_ohe[np.arange(Y.shape[0]), Y.squeeze()] = 1

    return y_ohe


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**1,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_relu = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()
