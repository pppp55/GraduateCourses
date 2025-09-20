import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def load_mat(path, d=16):
    data = scipy.io.loadmat(path)['zip']
    size = data.shape[0]
    y = data[:, 0].astype('int')
    X = data[:, 1:].reshape(size, d, d)
    return X, y

def cal_intensity(X):
    """
    X: (n, d), input data
    return intensity: (n, 1)
    """
    n = X.shape[0]
    return np.mean(X.reshape(n, -1), 1, keepdims=True)

def cal_symmetry(X):
    """
    X: (n, d), input data
    return symmetry: (n, 1)
    """
    n, d = X.shape[:2]
    Xl = X[:, :, :int(d/2)]
    Xr = np.flip(X[:, :, int(d/2):], -1)
    abs_diff = np.abs(Xl-Xr)
    return np.mean(abs_diff.reshape(n, -1), 1, keepdims=True)

def cal_feature(data):
    intensity = cal_intensity(data)
    symmetry = cal_symmetry(data)
    feat = np.hstack([intensity, symmetry])

    return feat

def cal_feature_cls(data, label, cls_A=1, cls_B=5):
    """ calculate the intensity and symmetry feature of given classes
    Input:
        data: (n, d1, d2), the image data matrix
        label: (n, ), corresponding label
        cls_A: int, the first digit class
        cls_B: int, the second digit class
    Output:
        X: (n', 2), the intensity and symmetry feature corresponding to 
            class A and class B, where n'= cls_A# + cls_B#.
        y: (n', ), the corresponding label {-1, 1}. 1 stands for class A, 
            -1 stands for class B.
    """
    feat = cal_feature(data)
    indices = (label==cls_A) + (label==cls_B)
    X, y = feat[indices], label[indices]
    ind_A, ind_B = y==cls_A, y==cls_B
    y[ind_A] = 1
    y[ind_B] = -1

    return X, y

def plot_feature(feature, y, plot_num, ax=None, classes=np.arange(10)):
    """plot the feature of different classes
    Input:
        feature: (n, 2), the feature matrix.
        y: (n, ) corresponding label.
        plot_num: int, number of samples for each class to be plotted.
        ax: matplotlib.axes.Axes, the axes to be plotted on.
        classes: array(0-9), classes to be plotted.
    Output:
        ax: matplotlib.axes.Axes, plotted axes.
    """
    cls_features = [feature[y==i] for i in classes]

    marks = ['s', 'o', 'D', 'v', 'p', 'h', '+', 'x', '<', '>']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'cyan', 'orange', 'purple']
    if ax is None:
        _, ax = plt.subplots()
    for i, feat in zip(classes, cls_features):
        ax.scatter(*feat[:plot_num].T, marker=marks[i], color=colors[i], label=str(i))
    plt.legend(loc='upper right')
    plt.xlabel('intensity')
    plt.ylabel('symmetry')
    return ax

def cal_error(theta, X, y, thres=1e-4):
    """calculate the binary error of the model w given data (X, y)
    theta: (d+1, 1), the weight vector
    X: (n, d), the data matrix [X, y]
    y: (n, ), the corresponding label
    """
    out = X @ theta - thres
    pred = np.sign(out)
    err = np.mean(pred.squeeze()!=y)
    return err

# prepare data
train_data, train_label = load_mat('train_data.mat') # train_data: (7291, 16, 16), train_label: (7291, )
test_data, test_label = load_mat('test_data.mat') # test_data: (2007, 16, 16), train_label: (2007, )

cls_A, cls_B = 1, 6
X, y, = cal_feature_cls(train_data, train_label, cls_A=cls_A, cls_B=cls_B)
X_test, y_test = cal_feature_cls(test_data, test_label, cls_A=cls_A, cls_B=cls_B)

# train_feat = cal_feature(train_data)
# plot_feature(train_feat, train_label, plot_num)
# plt.show()

# train
iters = 2000
d = 2
num_sample = X.shape[0]
threshold = 1e-4
theta = np.zeros((d, 1))

for iterate in range(iters):
    pass # TODO: add training code for perceptron and pocket

# plot Er_in and Er_out
# TODO

# plot decision boundary
# TODO you may utilize the plot_feature() function.