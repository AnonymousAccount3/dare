import os
import shutil
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from joblib import Parallel, delayed
from networks import DenseNet, LeNet, ResNet32
from utils import save_weights, load_weights, ModelCheckpoint, AUROC
from utils import MeanSquaredErrorClassif, GaussianNegativeLogLikelihood
from utils import mu_sigma, mean_squared_entropy, kl, entropy, disagreement, softmax
from utils import results_classification, results_regression
from methods import AnchoredNetwork, BaseEnsemble, DeepEnsemble, MOD, RDE, NegativeCorrelation, DARE
from datasets import reg1d, classif2d, citycam_weather, citycam_bigbus, citycam_cameras, cifar10, fashionmnist, mnist, cifar10_mnist, cifar10_resnet, cifar100_resnet, svhn_resnet

PARALLEL_METHOD = ["DeepEnsemble", "DeepEnsembleMSE", "AnchoredNetwork", "DARE"]
CLASSIFICATION_DATASET = ["2d_classif", "cifar10", "fashionmnist", "cifar10_resnet"]

base_dict = {
    "AnchoredNetwork": AnchoredNetwork,
    "BaseEnsemble": BaseEnsemble,
    "DeepEnsemble": DeepEnsemble,
    "DeepEnsembleMSE": DeepEnsemble,
    "MOD": MOD,
    "RDE": RDE,
    "NegativeCorrelation": NegativeCorrelation,
    "DARE": DARE
}

network_dict = {
    "DenseNet": DenseNet,
    "LeNet": LeNet,
    "ResNet32": ResNet32
}

optimizer_dict = {
    "SGD": SGD,
    "Adam": Adam
}

schedule_dict = {
    "PiecewiseConstantDecay": PiecewiseConstantDecay,
}

loss_dict = {
    "SparseCategoricalCrossentropy": SparseCategoricalCrossentropy,
    "MeanSquaredError": MeanSquaredError,
    "BinaryCrossentropy": BinaryCrossentropy,
    "MeanSquaredErrorClassif": MeanSquaredErrorClassif,
    "GaussianNegativeLogLikelihood": GaussianNegativeLogLikelihood
}

dataset_dict = {
    "2d_classif": classif2d,
    "1d_reg": reg1d,
    "citycam_weather": citycam_weather,
    "citycam_cameras": citycam_cameras,
    "citycam_bigbus": citycam_bigbus,
    "cifar10": cifar10,
    "fashionmnist": fashionmnist,
    "mnist": mnist,
    "cifar10_mnist": cifar10_mnist,
    "cifar10_resnet": cifar10_resnet,
    "cifar100_resnet": cifar100_resnet,
    "svhn_resnet": svhn_resnet
}

score_dict = {
    "mean_squared_entropy": mean_squared_entropy,
    "kl": kl,
    "disagreement": disagreement,
    "entropy": entropy,
}

callbacks_dict = {
    "ModelCheckpoint": ModelCheckpoint,
    "AUROC": AUROC,
}


def freeze_bn(model):
    for i in range(len(model.layers)):
        if model.layers[i].__class__.__name__ == "BatchNormalization":
            model.layers[i].trainable = False
    return model


def build_model(config):
    
    metrics = config.get("metrics", None)
    optimizer = config.get("optimizer", "SGD")
    schedule = config.get("schedule", None)
    
    n_estimators = config.get("n_estimators", 1)
    method = config.get("method")
    network = config.get("network")
    loss = config.get("loss")
    
    optimizer_params = config.get("optimizer_params", {})
    network_params = config.get("network_params", {})
    schedule_params = config.get("schedule_params", {})
    method_params = config.get("params", {})
    
    if schedule is not None:
        schedule = schedule_dict[schedule](**schedule_params)
        optimizer_params.pop("lr", None)
        optimizer_params.pop("learning_rate", None)
        optimizer = optimizer_dict[optimizer](schedule, **optimizer_params)
    else:
        optimizer = optimizer_dict[optimizer](**optimizer_params)
        
    if loss in ["SparseCategoricalCrossentropy", "BinaryCrossentropy"]:
        loss = loss_dict[loss](from_logits=True)
    else:
        loss = loss_dict[loss]()
    
    if method in PARALLEL_METHOD:
        network = network_dict[network](**network_params)
        model = base_dict[method](network.inputs, network.outputs, **method_params)
    else:
        method_params["n_estimators"] = n_estimators
        model = base_dict[method](network_dict[network], **method_params, **network_params)
        
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def fit_network(num, state, data, config):
    np.random.seed(state)
    tf.random.set_seed(state)
    
    x, y, xval, yval, xtest, ytest, xood, yood = data
    
    model = build_model(config)
    
    save_path = config.get("save_path", "")
    epochs = config.get("epochs", 1)
    batch_size = config.get("batch_size", 1)
    steps_per_epoch = config.get("steps_per_epoch", 1)
    verbose = config.get("verbose", 1)
    callbacks = config.get("callbacks", None)
    callbacks_params = config.get("callbacks_params", {})
    state = config.get("state", 0)
    method = config.get("method")
    finetune = config.get("finetune", None)
    
    callbaks_list = []
    if callbacks is not None:
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
            callbacks_params = [callbacks_params]
        for callback, params in zip(callbacks, callbacks_params):
            if callback == "AUROC":
                if isinstance(xood, list):
                    xood_auroc = dataset_dict[xood[0]]()
                else:
                    xood_auroc = xood
                callbaks_list.append(AUROC(tf.data.Dataset.from_tensor_slices(xtest).batch(batch_size),
                                           tf.data.Dataset.from_tensor_slices(xood_auroc).batch(batch_size)))
            else:
                callbaks_list.append(callbacks_dict[callback](**params))
    
    if finetune is not None:
        model = freeze_bn(model)
        model.load_weights(os.path.join(finetune + "_%i"%state, "net_%i.hdf5"%num))
    
    model.fit(x, y, epochs=epochs, batch_size=batch_size,
              validation_data=(xval, yval), callbacks=callbaks_list)
    
    save_weights(model, os.path.join(save_path, method + "_%i"%state), state=state, num_model=num)


def train(config):
    
    for k, v in config.items():
        print(k+": ", v)
    
    state = config.get("state", 0)
    dataset = config.get("dataset")
    method = config.get("method")
    n_jobs = config.get("n_jobs", None)
    n_estimators = config.get("n_estimators", 1)
    params = config.get("params", {})
    save_path = config.get("save_path", "")
    network = config.get("network")
    network_params = config.get("network_params", {})
    batch_size = config.get("batch_size", 1)
    ood_score_func = config.get("ood_score_func", None)
    
    data = dataset_dict[dataset](state)
    x, y, xval, yval, xtest, ytest, xood, yood = data
    
    input_shape = data[0].shape[1:]
    network_params["input_shape"] = input_shape
    config["network_params"] = network_params
    
    if not isinstance(params, list):
        params = [params]
    
    np.random.seed(state)
    states = np.random.choice(2**16, n_estimators)
    
    best_score = np.inf
    for p in params:
        config_p = copy.deepcopy(config)
        config_p["params"] = p
        config_p["save_path"] = "temp/"
        if method in PARALLEL_METHOD:
            if n_jobs is None:
                for s, num in zip(states, range(n_estimators)):
                    fit_network(num, s, data, config_p)
            else:
                Parallel(n_jobs=n_jobs)(delayed(fit_network)(num, s, data, config_p)
                                        for s, num in zip(states, range(n_estimators)))
        else:
            fit_network(0, states[0], data, config_p)
        
        model = load_weights(os.path.join("temp/", method + "_%i"%state), network_dict[network], **network_params)
        shutil.rmtree(os.path.join("temp/", method + "_%i"%state))
        ypred = model.predict(xval, batch_size=batch_size, verbose=1)
        if dataset in CLASSIFICATION_DATASET:
            ypred = softmax(ypred)
            if ypred.ndim < 3:
                ypred = ypred[:, :, np.newaxis]
            ypred = ypred.mean(-1)
            score = SparseCategoricalCrossentropy()(tf.identity(yval),
                                                    tf.identity(ypred)).numpy().mean()
        else:
            y_pred_mu, y_pred_sigma = mu_sigma(ypred)
            ypred = np.concatenate((y_pred_mu.reshape(-1, 1),
                                    np.log(y_pred_sigma.reshape(-1, 1)**2 + 1e-16)), axis=1)
            print(ypred[0, 0].dtype, yval[0].dtype)
            score = GaussianNegativeLogLikelihood()(tf.identity(yval),
                        tf.identity(ypred)).numpy().mean()
        print(method, str(p), "Score: %.4f"%score)
        if score <= best_score:
            save_weights(model, os.path.join(save_path, method + "_%i"%state), state)
            best_score = score
            config["best_params"] = p
            config["best_score"] = score
        else:
            if method == "AnchoredNetwork" and p["sigma"] != 10.:
                pass
            else:
                break
            
    model = load_weights(os.path.join(save_path, method + "_%i"%state), network_dict[network], **network_params)
    model.compile(loss=loss_dict[config.get("loss")]())
    val_metrics = model.evaluate(xval, yval, batch_size=batch_size, verbose=1, return_dict=True)
    config["val_loss"] = val_metrics["loss"]
    
    if dataset in CLASSIFICATION_DATASET:
        if not isinstance(xood, list):
            scores = results_classification(model, data, score_dict[ood_score_func])
        else:
            scores = {}
            for name in xood:
                xo = dataset_dict[name](state)
                if isinstance(xo, tuple):
                    xo = xo[4]
                print(name, xo.shape)
                scores_xo = results_classification(model,
                            (x, y, xval, yval, xtest, ytest, xo, None),
                            score_dict[ood_score_func])
                for k, v in scores_xo.items():
                    if "ood" in k:
                        scores[k + "_%s"%name] = v
                    else:
                        scores[k] = v
    else:
        scores = results_regression(model, data)
            
    logs_save_path = os.path.join("logs", save_path, method + "_%i.csv"%state)
    os.makedirs(os.path.dirname(logs_save_path), exist_ok=True)
    config.update(scores)
    pd.Series(config).to_csv(logs_save_path)