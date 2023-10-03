import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

EPS = np.finfo(np.float32).eps

class DARE(Model):
    
    def __init__(self,
                 inputs,
                 outputs,
                 threshold=0.):
        
        super().__init__(inputs, outputs)
        self.threshold = threshold
       
    
    def train_step(self, data):
        x, y = data
        
        if len(y.shape) < 2:
            y = tf.reshape(y, (-1, 1))
        
        with tf.GradientTape() as tape:
            
            y_pred = self(x, training=True)
            task_loss = tf.reduce_mean(self.compiled_loss(y, y_pred))
            
            weight_loss = 0.
            count = 0.
            for i in range(len(self.trainable_variables)):
                weight_loss += tf.reduce_sum(tf.math.log(EPS + tf.square(self.trainable_variables[i])))
                count += tf.reduce_sum(tf.ones_like(self.trainable_variables[i]))

            weight_loss /= count
            
            bool_ = tf.cast(tf.greater_equal(self.threshold, task_loss), weight_loss.dtype)
            loss = task_loss
            loss += -weight_loss * bool_
            loss += sum(self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        logs = {m.name: m.result() for m in self.metrics}
        logs["loss"] = task_loss
        logs["weight"] = weight_loss
        logs["lambda"] = bool_
        return logs
