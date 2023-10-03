import os
import gzip
import pickle
import numpy as np
import pandas as pd
import scipy
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import tensorflow as tf
from networks import ResNet32


def classif2d(state=None):
    
    np.random.seed(0)
    x, y = make_moons(n_samples=200, noise=0.1)
    
    x = x - np.array([0.5, 0.25])
    
    xval, yval = make_moons(n_samples=50, noise=0.1)
    xval = xval - np.array([0.5, 0.25])
    
    x_min, y_min = -3., -3
    x_max, y_max = 3, 3
    x_grid, y_grid = np.meshgrid(np.linspace(x_min-0.1, x_max+0.1, 100),
                                 np.linspace(y_min-0.1, y_max+0.1, 100))
    X_grid = np.stack([x_grid.ravel(), y_grid.ravel()], -1)
    
    return x, y, xval, yval, xval, yval, X_grid, None


def reg1d(state=None):
    
    np.random.seed(123)
    def f(x):
        return 0.3 * x + 0.3 * np.sin(2*np.pi*x) + 0.3 * np.sin(4*np.pi*x)

    Xt = np.linspace(-1.5, 1.5, 1000)
    yt = f(Xt)

    Xs = np.random.randn(50)*0.1 - 0.5
    Xs = np.concatenate((Xs, np.random.randn(50)*0.1 + 0.75))
    ys = f(Xs) + 0.02 * np.random.randn(Xs.shape[0])
    
    Xval = np.random.randn(10)*0.1 - 0.5
    Xval = np.concatenate((Xval, np.random.randn(10)*0.1 + 0.75))
    yval = f(Xval) + 0.02 * np.random.randn(Xval.shape[0])
    
    ys = ys.astype(np.float32)
    yval = yval.astype(np.float32)
    yt = yt.astype(np.float32)
    
    return Xs.reshape(-1, 1), ys, Xval.reshape(-1, 1), yval, Xval.reshape(-1, 1), yval, Xt.reshape(-1, 1), yt
    
    
def load_citycam(domains, path="./datasets/citycam"):
    indexes = []
    for domain in domains:
        path_dom = os.path.join(path, domain)
        for r, d, f in os.walk(path_dom):
            if domain[:6] == "bigbus":
                d = [""]
            for direct in d:
                if "checkpoints" not in direct:
                    x_path = os.path.join(path_dom, direct, "X.npy")
                    y_path = os.path.join(path_dom, direct, "y.npy")
                    time_path = os.path.join(path_dom, direct, "time.npy")

                    Xi = np.load(x_path)
                    yi = np.load(y_path)
                    time_i = np.load(time_path)
                    
                    if len(yi) != len(Xi):
                        print(len(yi), len(Xi))
                        
                    indexes += [domain] * len(yi)

                    try:
                        X = np.concatenate((X, Xi))
                        y = np.concatenate((y, yi))
                        time = np.concatenate((time, time_i))
                    except:
                        X = np.copy(Xi)
                        y = np.copy(yi)
                        time = np.copy(time_i)
    return X, y, time, np.array(indexes)
    
    
def get_day(x):
    return x.split(" ")[0]
    
    
def citycam_weather(state=None, return_indexes=False):
    cameras = ["164", "166", "572"]
    X, y, time, indexes = load_citycam(cameras)
    day = np.array(list(map(get_day, time)))
    
    X = X[day == "2016/02/23"]
    y = y[day == "2016/02/23"]
    time = time[day == "2016/02/23"]
    indexes = indexes[day == "2016/02/23"]
    
    y = y.astype(np.float32)

    id_index = np.argwhere(pd.to_datetime(time) < pd.to_datetime('2016-02-23 14:00:00')).ravel()
    ood_index = np.argwhere(pd.to_datetime(time) >= pd.to_datetime('2016-02-23 14:00:00')).ravel()

    np.random.seed(state)
    train_index, test_index = train_test_split(id_index, train_size=0.9)
    train_index, val_index = train_test_split(train_index, train_size=0.95)
    
    out = []
    index_list = []
    time_list = []
    for ind in [train_index, val_index, test_index, ood_index]:
        out.append(X[ind])
        out.append(y[ind])
        index_list.append(indexes[ind])
        time_list.append(time[ind])
        
    if return_indexes:
        return tuple(out), index_list, time_list
    
    return tuple(out)


def citycam_cameras(state=None, return_indexes=False):
    CAMERAS = [928, 846, 691, 551, 511, 495, 410, 403, 398, 253]
    CAMERAS = [str(c) for c in CAMERAS]
    
    np.random.seed(state)
    cameras = list(np.random.choice(CAMERAS, 5, replace=False))
    new_cameras = list(set(CAMERAS) - set(cameras))
    
    X, y, _, _ = load_citycam(cameras)
    Xood, yood, _, _ = load_citycam(new_cameras)
    
    y = y.astype(np.float32)
    yood = yood.astype(np.float32)

    np.random.seed(state)
    train_index, test_index = train_test_split(np.arange(len(X)), train_size=0.9)
    train_index, val_index = train_test_split(train_index, train_size=0.95)
    
    out = []
    for ind in [train_index, val_index, test_index]:
        out.append(X[ind])
        out.append(y[ind])
    out.append(Xood)
    out.append(yood)
    
    return tuple(out)


def citycam_bigbus(state=None, return_indexes=False):
    cameras = ["398", "403", "410", "495"]
    cameras_bigbus = ["bigbus/398-big_bus-mask1", "bigbus/398-big_bus-mask2",
                      "bigbus/bigbus-403", "bigbus/410-big_bus", "bigbus/bigbus-495"]
    
    X, y, _, _ = load_citycam(cameras)
    Xood, yood, _, _ = load_citycam(cameras_bigbus)
    
    y = y.astype(np.float32)
    yood = yood.astype(np.float32)

    np.random.seed(state)
    train_index, test_index = train_test_split(np.arange(len(X)), train_size=0.9)
    train_index, val_index = train_test_split(train_index, train_size=0.95)
    
    out = []
    for ind in [train_index, val_index, test_index]:
        out.append(X[ind])
        out.append(y[ind])
    out.append(Xood)
    out.append(yood)
    
    return tuple(out)
    
    
def cifar10(state=None):
    data = tf.keras.datasets.cifar10.load_data()
    
    x = data[0][0]
    y = data[0][1]
    xtest = data[1][0]
    ytest = data[1][1]
    
    np.random.seed(state)
    train_index, val_index = train_test_split(np.arange(len(x)), test_size=1000)
    
    return x[train_index], y[train_index], x[val_index], y[val_index], xtest, ytest, x[val_index], None
    
    
def fashionmnist(state=None):
    data = tf.keras.datasets.fashion_mnist.load_data()

    x = data[0][0]
    y = data[0][1]
    xtest = data[1][0]
    ytest = data[1][1]
    
    np.random.seed(state)
    train_index, val_index = train_test_split(np.arange(len(x)), test_size=1000)
    
    return x[train_index], y[train_index], x[val_index], y[val_index], xtest, ytest, ["mnist", "cifar10_mnist"], None


def mnist(state=None):
    data = tf.keras.datasets.mnist.load_data()
    xood = data[1][0]
    return xood


def cifar10_mnist(state=None):
    data = tf.keras.datasets.cifar10.load_data()
    xood = data[1][0]
    return xood[:, :28, :28, :].mean(-1)


def cifar10_resnet(state=None, path="./results/cifar10/"):   
    data = tf.keras.datasets.cifar10.load_data()
    
    x = data[0][0]
    y = data[0][1]
    xtest = data[1][0]
    ytest = data[1][1]

    resnet = ResNet32()
    resnet.load_weights(os.path.join(path, "DeepEnsemble_%i"%state, "net_0.hdf5"))
    model = tf.keras.Model(resnet.inputs, resnet.layers[-1].input)

    x = model.predict(x, batch_size=128, verbose=1)
    xtest = model.predict(xtest, batch_size=128, verbose=1)
    
    np.random.seed(state)
    train_index, val_index = train_test_split(np.arange(len(x)), test_size=1000)
    
    return x[train_index], y[train_index], x[val_index], y[val_index], xtest, ytest, ["cifar100_resnet", "svhn_resnet"], None


def cifar100_resnet(state=None, path="./results/cifar10/"):
    data = tf.keras.datasets.cifar100.load_data()
    xood = data[1][0]

    resnet = ResNet32()
    resnet.load_weights(os.path.join(path, "DeepEnsemble_%i"%state, "net_0.hdf5"))
    model = tf.keras.Model(resnet.inputs, resnet.layers[-1].input)

    xood = model.predict(xood, batch_size=128, verbose=1)

    return xood


def svhn_resnet(state=None, path="./results/cifar10/"):
    mat = scipy.io.loadmat('./datasets/svhn/test_32x32.mat')
    xood = mat["X"].transpose(3,0,1,2)

    resnet = ResNet32()
    resnet.load_weights(os.path.join(path, "DeepEnsemble_%i"%state, "net_0.hdf5"))
    model = tf.keras.Model(resnet.inputs, resnet.layers[-1].input)

    xood = model.predict(xood, batch_size=128, verbose=1)
    return xood