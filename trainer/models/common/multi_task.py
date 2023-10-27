import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs


class MMOELayer(tf.keras.layers.Layer):
    """A wrapper for basic dense layers, including Dense, BN, DropOut and l2

    Input shape
      - 2D tensor with shape: ``(batch_size, embedding_size)``.

    Output shape
      - 3D tensor with shape: ``(batch_size, gate_num, last_dense_layer_size)``.

    """

    def __init__(
        self,
        expert_num=2,
        gate_num=2,
        layer_sizes=(256, 128, 64),
        activation="relu",
        use_bn=True,
        dropout=0.0,
        l2=0.0001,
        **kwargs
    ):
        super(MMOELayer, self).__init__(**kwargs)
        self.experts = []
        for _ in range(expert_num):
            expert = tf.keras.Sequential(name="expert")
            for layer_size in layer_sizes:
                expert.add(
                    tf.keras.layers.Dense(
                        layer_size,
                        activation=activation,
                        kernel_regularizer=tf.keras.regularizers.l2(l2),
                    )
                )
            if use_bn:
                expert.add(tf.keras.layers.BatchNormalization())
            if dropout:
                expert.add(tf.keras.layers.Dropout(dropout))
            self.experts.append(expert)
        self.gates = [
            tf.keras.layers.Dense(expert_num, activation="softmax")
            for _ in range(gate_num)
        ]

    def call(self, inputs, **kwargs):
        # shape [batch_size, expert_num, last_dense_layer_size]
        experts_output = tf.stack(
            [expert(inputs, **kwargs) for expert in self.experts], axis=1
        )
        # shape [batch_size, gate_num, expert_num]
        gate_output = tf.stack([gate(inputs, **kwargs) for gate in self.gates], axis=1)
        # shape [batch_size, gate_num, last_dense_layer_size]
        return tf.matmul(gate_output, experts_output)


class UncertaintyWeightingLayer(tf.keras.layers.Layer):
    """Weight based on task-depent uncertainty

    Input shape
      - list of task number size.

    Output shape
      - scalar value.

    References
        - [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115.pdf)
    """

    def __init__(self, task_num, **kwargs):
        super(UncertaintyWeightingLayer, self).__init__(**kwargs)
        self.task_num = task_num
        # defined as the log(σ^2) for numerical stability
        self.uncertainty_weights = [
            tf.Variable(initial_value=1 / task_num, name="uncertainty_weight_" + str(i))
            for i in range(task_num)
        ]

    def call(self, loss_list, **kwargs):
        assert len(loss_list) == self.task_num, "loss number must equal to task number"

        final_loss = []
        for i in range(self.task_num):
            final_loss.append(
                tf.math.multiply(
                    tf.math.exp(-self.uncertainty_weights[i]),
                    loss_list[i],
                )
                + self.uncertainty_weights[i]
            )
        return tf.reduce_sum(final_loss, axis=0)


class GradientNormModel(tfrs.Model):
    """customized training step based on gradient norm

    References
        - [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://arxiv.org/pdf/1711.02257v4.pdf)
    """

    def __init__(self, hparams, *args, **kwargs):
        super(GradientNormModel, self).__init__(*args, **kwargs)
        self.task_num = hparams.task_num
        self.loss0 = [
            tf.Variable(0, trainable=False, dtype=tf.float32)
            for _ in range(self.task_num)
        ]
        self.alpha = hparams.alpha
        self.loss_weights = [
            tf.Variable(initial_value=1 / self.task_num, name="loss_weight_" + str(i))
            for i in range(self.task_num)
        ]
        self.gn_optimizer = tf.keras.optimizers.Adam(hparams.learning_rate)

    def train_step(self, inputs):
        """Custom train step using the `compute_loss` method."""

        # the tape need to be persistent if we need to use it multiple times
        with tf.GradientTape(persistent=True) as tape:
            loss_list = self.compute_loss(inputs, training=True)
            assert (
                len(loss_list) == self.task_num
            ), "loss number must equal to task number"

            # a trick to only keep the first loss value
            for idx, loss in enumerate(loss_list):
                tf.cond(
                    tf.equal(self.loss0[idx], 0.0),
                    lambda: self.loss0[idx].assign(loss),
                    lambda: loss,
                )

            # the gradient norms
            # shape [task_num, second_last_layer_size*last_layer_size]
            # trainable_variables[0] means only use the weights, ignore bias
            gradient_norm = tf.reshape(
                [
                    tf.math.l2_normalize(
                        tape.gradient(
                            self.loss_weights[i] * loss_list[i],
                            self.last_shared_layer.trainable_variables[0],
                        ),
                        axis=-1,
                    )
                    for i in range(self.task_num)
                ],
                [self.task_num, -1],
            )
            # shape [1, second_last_layer_size*last_layer_size]
            # get the mean gradient value across all tasks
            gradient_norm_avg = tf.reduce_mean(gradient_norm, axis=0, keepdims=True)
            # shape [task_num]
            loss_ratio = [loss_list[i] / self.loss0[i] for i in range(self.task_num)]
            loss_ratio_avg = tf.reduce_mean(loss_ratio)

            # shape [task_num, 1]
            # the relative inverse training rate, it will be broadcasted to gradient_norm_avg
            train_rate = tf.stack([l / loss_ratio_avg for l in loss_ratio])[
                :, tf.newaxis
            ]

            regularization_loss = sum(self.losses)

            # shape [task_num, second_last_layer_size*last_layer_size]
            # gradient normed loss
            loss_grad = tf.math.abs(
                gradient_norm - gradient_norm_avg * tf.math.pow(train_rate, self.alpha)
            )
            loss_grad = tf.reduce_sum(loss_grad) + regularization_loss

            # update weights
            weight_grad = tape.gradient(loss_grad, self.loss_weights)
            self.gn_optimizer.apply_gradients(zip(weight_grad, self.loss_weights))

            # total loss
            loss = tf.reduce_sum(
                [self.loss_weights[i] * loss_list[i] for i in range(self.task_num)]
            )

            total_loss = loss + regularization_loss

        # skip weights
        trainable_variables = self.trainable_variables
        for w in self.loss_weights:
            trainable_variables.remove(w)
        gradients = tape.gradient(total_loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        del tape
        return metrics

    def test_step(self, inputs):
        """Custom test step using the `compute_loss` method."""

        loss_list = self.compute_loss(inputs, training=False)
        loss = tf.reduce_sum(
            [self.loss_weights[i] * loss_list[i] for i in range(self.task_num)]
        )
        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics


class DynamicWeightAveragingModel(tfrs.Model):
    """customized training step based on dynamic weight averaging

    References
        - [End-to-End Multi-Task Learning with Attention](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf)
    """

    def __init__(self, hparams, *args, **kwargs):
        super(DynamicWeightAveragingModel, self).__init__(*args, **kwargs)
        self.task_num = hparams.task_num
        self.temperature = hparams.temperature
        self.prev_loss = [
            tf.Variable(0, trainable=False, dtype=tf.float32)
            for _ in range(self.task_num)
        ]
        self.loss_weights = [
            tf.Variable(initial_value=1 / self.task_num, name="loss_weight_" + str(i))
            for i in range(self.task_num)
        ]

    def train_step(self, inputs):
        """Custom train step using the `compute_loss` method."""

        with tf.GradientTape() as tape:
            loss_list = self.compute_loss(inputs, training=True)
            assert (
                len(loss_list) == self.task_num
            ), "loss number must equal to task number"

            # a trick to initialize the previous loss for the first step
            for idx, loss in enumerate(loss_list):
                tf.cond(
                    tf.equal(self.prev_loss[idx], 0.0),
                    lambda: self.prev_loss[idx].assign(loss),
                    lambda: loss,
                )

            # shape: a list of task_num size
            # the inverse training rate
            train_rate = [
                loss_list[i] / self.prev_loss[i] / self.temperature
                for i in range(self.task_num)
            ]
            # unshape weight value from tensor to scalar
            weights = tf.split(tf.math.softmax(train_rate), self.task_num)
            for idx, weight in enumerate(weights):
                self.loss_weights[idx].assign(tf.reshape(weight, []))

            # keep the last loss
            for idx, loss in enumerate(loss_list):
                self.prev_loss[idx].assign(loss)

            # total loss
            loss = tf.reduce_sum(
                [self.loss_weights[i] * loss_list[i] for i in range(self.task_num)]
            )
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, inputs):
        """Custom test step using the `compute_loss` method."""

        loss_list = self.compute_loss(inputs, training=False)
        loss = tf.reduce_sum(
            [self.loss_weights[i] * loss_list[i] for i in range(self.task_num)]
        )
        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics


class DynamicTaskPrioritizationLayer(tf.keras.layers.Layer):
    """Weight based on focal loss

    Input shape
      - list of kpis and losses

    Output shape
      - scalar value

    References
        - [Dynamic Task Prioritization for Multitask Learning](https://openaccess.thecvf.com/content_ECCV_2018/papers/Michelle_Guo_Focus_on_the_ECCV_2018_paper.pdf)
    """

    def __init__(self, task_num, gamma=1.0, **kwargs):
        super(DynamicTaskPrioritizationLayer, self).__init__(**kwargs)
        self.task_num = task_num
        self.gamma = gamma

    def call(self, kpi_list, loss_list, **kwargs):
        assert (
            len(kpi_list) == self.task_num and len(loss_list) == self.task_num
        ), "kpi and loss number must equal to task number"

        final_loss = []
        for i in range(self.task_num):
            kpi = kpi_list[i]
            loss = loss_list[i]
            final_loss.append(
                -tf.math.pow(1 - kpi, self.gamma) * tf.math.log(kpi) * loss
            )
        return tf.reduce_sum(final_loss, axis=0)


class PELTRModel(tfrs.Model):
    """customized training step based on PELTR

    References
        - [A Pareto-Efficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation](http://ofey.me/papers/Pareto.pdf)
    """

    def __init__(self, hparams, *args, **kwargs):
        super(PELTRModel, self).__init__(*args, **kwargs)
        self.task_num = hparams.task_num
        self.loss_weights = [
            tf.Variable(initial_value=1 / self.task_num, name="loss_weight_" + str(i))
            for i in range(self.task_num)
        ]

    def pareto_step(self, weights_list, out_gradients_list):
        # reference https://github.com/jackielinxiao/PE-LTR/blob/master/main.py
        model_gradients = out_gradients_list
        M1 = np.matmul(model_gradients, np.transpose(model_gradients))
        e = np.mat(np.ones(np.shape(weights_list)))
        M = np.hstack((M1, np.transpose(e)))
        mid = np.hstack((e, np.mat(np.zeros((1, 1)))))
        M = np.vstack((M, mid))
        z = np.mat(np.zeros(np.shape(weights_list)))
        nid = np.hstack((z, np.mat(np.ones((1, 1)))))
        w = np.matmul(
            np.matmul(M, np.linalg.inv(np.matmul(M, np.transpose(M)))),
            np.transpose(nid),
        )
        if len(w) > 1:
            w = np.transpose(w)
            w = w[0, 0 : np.shape(w)[1]]
            mid = np.where(w > 0, 1.0, 0)
            nid = np.multiply(mid, w)
            uid = sorted(nid[0].tolist()[0], reverse=True)
            sv = np.cumsum(uid)
            rho = np.where(uid > (sv - 1.0) / range(1, len(uid) + 1), 1.0, 0.0)
            r = max(np.argwhere(rho))
            theta = max(0, (sv[r] - 1.0) / (r + 1))
            w = np.where(nid - theta > 0.0, nid - theta, 0)
        return w

    def train_step(self, inputs):
        """Custom train step using the `compute_loss` method."""

        with tf.GradientTape(persistent=True) as tape:
            loss_list = self.compute_loss(inputs, training=True)
            assert (
                len(loss_list) == self.task_num
            ), "loss number must equal to task number"

            # total loss
            loss = tf.reduce_sum(
                [self.loss_weights[i] * loss_list[i] for i in range(self.task_num)]
            )
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        # the gradients for each loss
        # shape [task_num, second_last_layer_size * last_layer_size]
        # trainable_variables[0] means only use the weights, ignore bias
        loss_gradients = tf.reshape(
            [
                tape.gradient(
                    loss_list[i],
                    self.last_shared_layer.trainable_variables[0],
                )
                for i in range(self.task_num)
            ],
            [self.task_num, -1],
        )

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # update weights
        # shape [1, task_num]
        loss_weights = tf.expand_dims(tf.stack(self.loss_weights), axis=0)
        mid = tf.py_function(
            func=self.pareto_step, inp=[loss_weights, loss_gradients], Tout=tf.float32
        )
        for i in range(self.task_num):
            self.loss_weights[i].assign(mid[0][i])

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        del tape
        return metrics

    def test_step(self, inputs):
        """Custom test step using the `compute_loss` method."""

        loss_list = self.compute_loss(inputs, training=False)
        loss = tf.reduce_sum(
            [self.loss_weights[i] * loss_list[i] for i in range(self.task_num)]
        )
        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics
