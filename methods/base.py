import tensorflow as tf
from tensorflow.keras import Model


def fit_network(net, x, y, **fit_params):
    net.fit(x, y, **fit_params)


class BaseEnsemble(Model):

    def __init__(self,
                 build_fn,
                 n_estimators=5,
                 lambda_=1.,
                 **params):
        
        super().__init__()
        self.n_estimators = n_estimators
        self.lambda_ = lambda_
        
        for i in range(self.n_estimators):
            setattr(self, "network_%i"%i, build_fn(**params))
            
        for i in range(self.n_estimators):
            getattr(self, "network_%i"%i)._name = "network_%i"%i
            
            
    def call(self, inputs, training=False):
        preds = []
        for i in range(self.n_estimators):
            preds.append(getattr(self, "network_%i"%i)(inputs, training=training))
        return tf.stack(preds, axis=-1)
    
    
    def test_step(self, data):
        # Unpack the data
        x, y = data
        if len(y.shape) < 2:
            y = tf.reshape(y, (-1, 1))
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        for i in range(self.n_estimators):
            self.compiled_loss(y, y_pred[:, :, i])
        # Update the metrics.
        self.compiled_metrics.update_state(y, tf.reduce_mean(y_pred, axis=-1))
        # Return a dict mapping metric names to current value
        logs = {m.name: m.result() for m in self.metrics}
        return logs