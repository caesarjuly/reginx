import tensorflow as tf
import tensorflow_recommenders as tfrs


class WideAndDeepTFRS(tfrs.Model):
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(inputs, training=True)

            # Handle regularization losses as well.
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

            linear_vars = self.wide.trainable_variables
            dnn_vars = self.deep.trainable_variables
            linear_grads, dnn_grads = tape.gradient(loss, (linear_vars, dnn_vars))

            linear_optimizer = self.optimizer[0]
            dnn_optimizer = self.optimizer[1]
            linear_optimizer.apply_gradients(zip(linear_grads, linear_vars))
            dnn_optimizer.apply_gradients(zip(dnn_grads, dnn_vars))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics
