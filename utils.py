import time
import os
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from methods import BaseEnsemble

EPS = np.finfo(float).eps


## Utils

def save_weights(model, dir_path, state=0, num_model=None):
    os.makedirs(dir_path, exist_ok=True)
    if isinstance(model, BaseEnsemble):
        for num in range(model.n_estimators):
            file_path = os.path.join(dir_path, "net_%i.hdf5"%num)
            getattr(model, "network_%i"%num).save_weights(file_path)
    else:
        if num_model is None:
            for num in range(0, 1000):
                file_path = os.path.join(dir_path, "net_%i.hdf5"%num)
                if not os.path.isfile(file_path):
                    break
        else:
            file_path = os.path.join(dir_path, "net_%i.hdf5"%num_model)
        model.save_weights(file_path)

        
def load_weights(path, build_fn, **kwargs):
    if ".hdf5" in path:
        model = build_fn(**kwargs)
        model.load_weights(path)
    else:
        for num in range(0, 1000):
            file_path = os.path.join(path, "net_%i.hdf5"%num)
            if not os.path.isfile(file_path):
                break
        model = BaseEnsemble(build_fn, n_estimators=num, **kwargs)
        for num in range(0, model.n_estimators):
            file_path = os.path.join(path, "net_%i.hdf5"%num)
            getattr(model, "network_%i"%num).load_weights(file_path)
    return model
        

## Callbacks

class ModelCheckpoint(tf.keras.callbacks.Callback):
    
    def __init__(self, path="",
                 monitor="val_loss", threshold=None):
        super().__init__()
        rand = np.random.choice(2**16)
        t = int(time.time()*1000)
        filename = path + "net_%i_%i.hdf5"%(rand, t)
        self.filename = filename
        self.threshold = threshold
        self.monitor = monitor
        self.best = np.inf
        self.last_save = 0
        
    def on_epoch_end(self, epoch, logs=None):
        monitor = logs[self.monitor]
        if self.threshold is None:
            if monitor <= self.best:
                self.model.save_weights(self.filename)
                print("Model saved! Epoch %i"%epoch)
                print("Best update! %.3f"%monitor)
                self.last_save = epoch
                self.best = monitor
        else:
            if monitor <= self.threshold:
                self.model.save_weights(self.filename)
                print("Model saved! Epoch %i"%epoch)
                self.last_save = epoch
                self.best = monitor
    
    def on_train_end(self, logs=None):
        if os.path.isfile(self.filename):
            print("Restore Weights : Epoch %i  Val loss %.3f"%(self.last_save, self.best))
            self.model.load_weights(self.filename)
            os.remove(self.filename)
        else:
            print("No model saved !")
        

class AUROC(tf.keras.callbacks.Callback):
    
    def __init__(self, xtest, xood):
        self.xtest = xtest
        self.xood = xood
        self.auroc = []
        
    def on_epoch_end(self, epoch, logs=None):
        yp_id = self.model.predict(self.xtest, verbose=1)
        yp_ood = self.model.predict(self.xood, verbose=1)
        
        in_dist = mean_squared_entropy(yp_id)
        ood_dist = mean_squared_entropy(yp_ood)

        auroc = auroc_score(in_dist, ood_dist)
        print("\n AUROC : %.4f \n"%auroc)
        self.auroc.append(auroc)
        pd.DataFrame(self.auroc).to_csv("%s_auroc.csv"%self.model.name)
        

## Loss

class MeanSquaredErrorClassif(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        num_classes = tf.shape(y_pred)[1]
        if num_classes != 1:
            y_true_ohe = tf.one_hot(tf.reshape(y_true, (-1,)), depth=num_classes)
            y_true_ohe *= tf.cast(num_classes, y_pred.dtype)
        else:
            y_true_ohe = tf.reshape(tf.cast(y_true, y_pred.dtype), tf.shape(y_pred))
        return tf.reduce_mean(tf.square(y_pred - y_true_ohe), axis=-1)


class GaussianNegativeLogLikelihood(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):        
        mu = tf.reshape(y_pred[:, 0], tf.shape(y_true))
        log_sigma_2 = tf.reshape(y_pred[:, 1], tf.shape(y_true))
        error = tf.square(y_true - mu)
        log_sigma_2 = tf.clip_by_value(log_sigma_2, -10., 10000.)
        inv_sigma_2 = tf.exp(-log_sigma_2)
        return 0.5*(inv_sigma_2*error + log_sigma_2)
        
## Metrics


def mean_squared_entropy(y):
    num_classes = y.shape[1]
    if y.shape[1] < 2:
        y = np.concatenate((1-y, y), axis=1)
    argmin = np.argmin(np.square(y-num_classes), axis=1)
    mask = np.zeros(y.shape)
    if y.ndim > 2:
        mask[np.arange(y.shape[0])[:, np.newaxis],
             argmin,
             np.arange(y.shape[2])[np.newaxis, :]] = num_classes * 1.
    else:
        mask[np.arange(y.shape[0]), argmin] = num_classes * 1.
    if y.ndim > 2:
        variance = y.var(-1).mean(1)
    else:
        variance = 0.
    return np.mean(np.square(y-mask), axis=(1, 2)) + variance
        
        
def softmax(y):
    if y.shape[1] == 1:
        y = 1/(1+np.exp(-y))
        y = np.concatenate((1-y, y), axis=1)
        return y
    else:
        return tf.keras.activations.softmax(tf.identity(y), axis=1).numpy()


def entropy(y):
    ys = softmax(y)
    if ys.ndim < 3:
        ys = ys[:, :, np.newaxis]
    ys = ys.mean(-1)
    return -np.sum(ys*np.log(ys+1e-16), 1)


def auroc_score(id_score, ood_score):
    gt = np.zeros(len(id_score) + len(ood_score))
    gt[len(id_score):] = 1
    y_pred = np.concatenate((id_score, ood_score))
    auc = roc_auc_score(gt, y_pred)
    return auc


def kl(x, y):
    return np.sum(x * np.log(x/(y+1e-16) + 1e-16), 1)


def disagreement(y):
    return np.sum(kl(y, y.mean(-1, keepdims=True)), axis=-1)


def mu_sigma(y):
    if y.shape[1] == 1:
        return y.mean(-1).ravel(), y.std(-1).ravel()
    else:
        if y.ndim < 3:
            y = y[:, :, np.newaxis]
        mu = y[:, 0, :]
        mean_mu = np.mean(mu, axis=-1)
        sigma2 = np.exp(np.clip(y[:, 1, :], -np.inf, 10.))
        mean_sigma = np.clip(np.mean(sigma2 + mu**2, axis=-1) - mean_mu**2, 0., np.inf)
        mean_sigma = np.sqrt(mean_sigma)
    return mean_mu, mean_sigma


def calibration_curve_regression(y_true, y_pred_mu, y_pred_sigma, n_bins=10):
    mu = y_pred_mu; sigma = y_pred_sigma
    alphas = np.linspace(1/(n_bins+1), 1, n_bins, endpoint=False)
    pred_alphas = []
    for alpha in alphas:
        length = norm.ppf(1-alpha/2., scale=sigma)
        in_interval = (y_true >= mu - length) & (y_true <= mu + length)
        pred_alphas.append(np.mean(in_interval))
    alphas = 1 - alphas
    pred_alphas = np.array(pred_alphas)
    return alphas[::-1], pred_alphas[::-1]


def calibration_curve_classification(y_true, y_pred, n_bins=10):
    cal_curve_x = []
    cal_curve_y = []
    if y_true.shape[1] == 1:
        cal_curve = calibration_curve(y_true.astype(float),
                                      y_pred, n_bins=10,
                                      strategy="quantile")
        cal_curve_x.append(cal_curve[0])
        cal_curve_y.append(cal_curve[1])
    else:
        for k in range(y_true.shape[1]):
            cal_curve = calibration_curve((y_true.argmax(1) == k).astype(float),
                                          y_pred[:, k], n_bins=10,
                                          strategy="quantile")
            cal_curve_x.append(cal_curve[0])
            cal_curve_y.append(cal_curve[1])
    return np.concatenate(cal_curve_x), np.concatenate(cal_curve_y)


def results_regression(model, data):
    scores = {}
    
    x, y, xval, yval, xtest, ytest, xood, yood = data
    yptest = model.predict(xtest, batch_size=256, verbose=1)
    ypood = model.predict(xood, batch_size=256, verbose=1)
    ypval = model.predict(xval, batch_size=256, verbose=1)
    
    mu_test, sigma_test = mu_sigma(yptest)
    
    print(mu_test.shape, sigma_test.shape, ytest.shape)
    
    mae_score = np.abs(ytest.ravel() - mu_test.ravel()).mean()
    yptest_mu_sigma = np.concatenate((mu_test.reshape(-1, 1),
                      np.log(sigma_test.reshape(-1, 1)**2 + 1e-16)), axis=1)
    
    print(yptest_mu_sigma.shape)
    
    nll_score = GaussianNegativeLogLikelihood()(tf.identity(ytest.ravel()),
                                tf.identity(yptest_mu_sigma)).numpy().mean()
    cal_curve = calibration_curve_regression(ytest.ravel(), mu_test, sigma_test)
    cal_score = np.mean(np.abs(cal_curve[0] - cal_curve[1]))
    
    print("Test: MAE %.4f  NLL %.4f  Cal %.4f"%(mae_score,
                                                nll_score,
                                                cal_score))

    scores["mae_id"] = mae_score
    scores["nll_id"] = nll_score
    scores["cal_id"] = cal_score
    
    mu_ood, sigma_ood = mu_sigma(ypood)
    mae_score = np.abs(yood.ravel() - mu_ood.ravel()).mean()
    ypood_mu_sigma = np.concatenate((mu_ood.reshape(-1, 1),
                      np.log(sigma_ood.reshape(-1, 1)**2 + 1e-16)), axis=1)
    nll_score = GaussianNegativeLogLikelihood()(tf.identity(yood.ravel()),
                tf.identity(ypood_mu_sigma)).numpy().mean()
    cal_curve = calibration_curve_regression(yood.ravel(), mu_ood, sigma_ood)
    cal_score = np.mean(np.abs(cal_curve[0] - cal_curve[1]))
    
    print("OOD:  MAE %.4f  NLL %.4f  Cal %.4f"%(mae_score,
                                                nll_score,
                                                cal_score))
    scores["mae_ood"] = mae_score
    scores["nll_ood"] = nll_score
    scores["cal_ood"] = cal_score

    mu_val, sigma_val = mu_sigma(ypval)
    error_val = np.abs(yval.ravel() - mu_val.ravel())
    conf_val = norm.ppf(0.95) * sigma_val
    offset = np.quantile(error_val - conf_val, q=0.95)

    cov_cali_test = np.mean(norm.ppf(0.95) * sigma_test + offset >=
                            (ytest.ravel() - mu_test.ravel()))
    cov_cali_ood = np.mean(norm.ppf(0.95) * sigma_ood + offset >=
                            (yood.ravel() - mu_ood.ravel()))

    scores["cov_test"] = cov_cali_test
    scores["cov_ood"] = cov_cali_ood

    print("Cov Test: %.4f  Cov OOD: %.4f"%(cov_cali_test, cov_cali_ood))
    
    return scores
    
    
def results_classification(model, data, ood_score_fun):
    scores = {}
    
    x, y, xval, yval, xtest, ytest, xood, yood = data
    
    yptest = model.predict(xtest, batch_size=256, verbose=1)
    ypood = model.predict(xood, batch_size=256, verbose=1)
    
    in_dist = ood_score_fun(yptest)
    ood_dist = ood_score_fun(ypood)
    
    auc = auroc_score(in_dist, ood_dist)
    fpr = 1 - np.mean(ood_dist >= np.quantile(in_dist, 0.95))
    
    yptest_soft = softmax(yptest)
    if yptest_soft.ndim < 3:
        yptest_soft = yptest_soft[:, :, np.newaxis]
    yptest_soft = yptest_soft.mean(-1)
    
    acc_score = np.mean(yptest_soft.argmax(1) == ytest)
    nll_test = tf.keras.losses.SparseCategoricalCrossentropy()(ytest,
                                                               yptest_soft).numpy().mean()
    
    print("Test: Acc %.4f  NLL: %.4f"%(acc_score, nll_test))
    
    print("OOD: AUC: %.4f  FPR: %.4f"%(auc, fpr))
    
    scores["acc_id"] = acc_score
    scores["nll_id"] = nll_test
    scores["auc_ood"] = auc
    scores["fpr_ood"] = fpr
    return scores