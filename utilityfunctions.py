import os
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def retrieve_mnist():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    y = y.astype(int)
    X = X / 255.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000/70000, random_state=42)

    return X_train, X_test, y_train, y_test



def save_model(model, directory, filename):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)

    joblib.dump(model, filepath)



def plot_decision_boundary(model, X, y, title):
    # Create a grid to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect('equal', adjustable='box')



def create_image_from_array(array, output_path=None):
    if len(array) != 28 * 28:
         raise ValueError("Input array must have exactly 784 elements.")

    image_array = np.reshape(array, (28, 28))

    image_array = (image_array * 255).astype(np.uint8)

    plt.imshow(image_array, cmap="gray")
    plt.axis("off")
    plt.show()

    if output_path:
        image = Image.fromarray(image_array)
        image.save(output_path)

    return image_array