import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def main():
    digits = load_digits()

    n_samples = len(digits.images)

    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = KNeighborsClassifier(n_neighbors=3)

    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    confusion_matrix(y_test, predicted)

    image_with_prediction = list(zip(digits.images, clf.predict(X)))

    for pos, (image, prediction) in enumerate(image_with_prediction[:20]):
        plt.subplot(4, 5, pos + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r)
        plt.title("Prediction: %i" % prediction)

    plt.show()


if __name__ == "__main__":
    main()
