import tensorflow as tf
from .base import BaseEnsemble

    
class MOD(BaseEnsemble): 


    def train_step(self, data):
        X, y = data
        
        if len(y.shape) < 2:
            y = tf.reshape(y, (-1, 1))
        
        X_ood = tf.random.uniform(
            tf.shape(X),
            minval=tf.cast(tf.reduce_min(X, axis=0), tf.float32),
            maxval=tf.cast(tf.reduce_max(X, axis=0), tf.float32),
            dtype=tf.float32)
        
        losses = []
        
        if hasattr(self.loss, "__name__") and self.loss.__name__ == "nll_loss":
            alpha = 1.
        else:
            alpha = 0.
        
        with tf.GradientTape() as tape:
            
            preds = self(X, training=True)
            preds_ood = self(X_ood, training=False)
            
            var_ood_nll = tf.math.reduce_variance(preds_ood[:, 0, :], axis=-1)
            var_ood_nll = tf.reduce_mean(var_ood_nll)
            
            var_ood = tf.math.reduce_variance(preds_ood, axis=-1)
            var_ood = tf.reduce_mean(var_ood)
            
            var_ood = alpha * var_ood_nll + (1-alpha) * var_ood
            
            loss = 0.
            for i in range(self.n_estimators):
                yp = preds[:, :, i]
                loss += self.compiled_loss(y, yp)
            loss /= self.n_estimators

            loss += -self.lambda_ * var_ood
            loss += sum(self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics
        self.compiled_metrics.update_state(y, tf.reduce_mean(preds, axis=-1))
        # Return a dict mapping metric names to current value
        logs = {m.name: m.result() for m in self.metrics}
        logs["reg"] = var_ood
        return logs