import numpy as np
import matplotlib.pyplot as plt


def generate_X(start, end):
    x = np.arange(start, end)  # making two high corelative components
    y = 10 * x + np.random.randn(69)*15
    X = np.vstack((x, y))
    return X


def visual(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.scatter(x, y)
    ax.legend()
    plt.show()


def centred(X, x, y):
    return (X[0] - x.mean(), X[1] - y.mean())  # moving our vales to centre


def make_cov_matrix(matrix):
    cov_matrix = np.cov(matrix)
    Variance_x = np.cov(matrix)[0, 0]  # variance of x and y are diagonal
    Variance_y = np.cov(matrix)[1, 1]  # elements
    volume_of_loss(Variance_x, Variance_y)
    # Covariance_xy = np.cov(X_centr)[0, 1]
    return cov_matrix


def volume_of_loss(Var_x, Var_y):
    print("large vector(%) ", Var_x / Var_y * 100)
    print("small vector(%) ", (1 - Var_x / Var_y) * 100)


def make_proection(covmarix, X_centr):
    _, eignvecor = np.linalg.eig(covmarix)  # finding eignvalue and eignvecor
    vector = -eignvecor[:, -1]
    X_pca = np.dot(vector, X_centr)  # Transposend vector * X_centr
    return X_pca


def PCA():
    X = generate_X(1, 70)
    X_centred = centred(X, X[0], X[1])
    # visual(X_centred[0], X_centred[1])
    cov_matrix = make_cov_matrix(X_centred)
    return make_proection(cov_matrix, X_centred)


print(PCA())
