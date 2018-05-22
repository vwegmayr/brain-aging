import tensorflow as tf


class NetworkHeadBase(object):
    """
    This is a generic network head.
    Can be a prediction head, classification head, adversarial netw head ...
    """

    def __init__(
        self,
        name,
        model,
        last_layer,
        features,
        # Arguments from config
        loss_weight_in_global_loss=None,
        train_only_globally=True,
        head_l1_regularization=None,
    ):
        """
        Argument:
            - loss_weight_in_global_loss:
                Multiplicative factor for the loss in the global network loss
            - train_only_globally:
                If True, the head is trained with the global loss
                Otherwise, there are 2 optimizations operations:
                    * One for the head only (loss and regularization)
                    * One global (loss * $loss_weight_in_global_loss)
            - head_l1_regularization:
                If not null, a L1 regularization term is added to loss
                    (either local loss, or global loss)
                Regularization only affects head variables (not network body)
        """
        # Sanity checks
        assert(hasattr(self, 'loss'))
        assert(hasattr(self, 'metrics'))
        # This would mean no training at all
        assert(not (
            loss_weight_in_global_loss is None and
            train_only_globally
        ))

        self.name = name
        self.loss_weight_in_global_loss = loss_weight_in_global_loss
        self.train_only_globally = train_only_globally
        self.head_l1_regularization = head_l1_regularization
        self.scope = tf.get_variable_scope().name
        self.trainable_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            tf.get_variable_scope().name,
        )
        with tf.variable_scope('regularization_loss'):
            self._set_regularization_loss()
        with tf.variable_scope('head_loss'):
            self.head_training_loss = self._get_head_training_loss()
        with tf.variable_scope('global_loss_contribution'):
            self.global_loss_contribution = self._get_global_loss_contribution()

        # Setup summary for metrics
        self.metrics.update({'loss': self.loss})
        # Metrics are averaged over all test set (see @get_evaluated_metrics)
        if model.is_training_bool:
            for name, value in self.metrics.items():
                tf.summary.scalar(name, value)

        with tf.variable_scope('evaluated_metrics'):
            self._evaluated_metrics = self._get_evaluated_metrics()

    # -------------------------- Evaluation/training metrics
    def get_logged_training_variables(self):
        train_variables = {}
        if self.l1_regularization_loss_raw is not None:
            train_variables.update({
                'loss_l1_regularization_loss': self.loss,
            })
        if self.head_training_loss is not None:
            train_variables.update({
                'head_training_loss': self.head_training_loss,
            })
        if self.global_loss_contribution is not None:
            train_variables.update({
                'global_loss_contribution': self.global_loss_contribution,
            })
        train_variables.update(self.metrics)
        return train_variables

    def get_evaluated_metrics(self):
        return self._evaluated_metrics

    def _get_evaluated_metrics(self):
        metric_ops = {
            name: tf.metrics.mean(value, name=name)
            for name, value in self.metrics.items()
        }
        return metric_ops

    def get_predictions(self):
        """
        Returns a dict {name: tensor} of predictions
        """
        return {}

    # -------------------------- Basic getters
    def get_global_loss_contribution(self):
        """
        Returns contribution to global loss
        """
        if self.global_loss_contribution is None:
            return 0
        return self.global_loss_contribution

    def get_head_train_op(self, optimizer, **kwargs):
        """
        Operation for training the head only
        """
        if self.head_training_loss is None:
            return tf.no_op()

        return optimizer.minimize(
            loss=self.head_training_loss,
            var_list=self.trainable_variables,
            **kwargs
        )

    def get_name(self):
        return self.name

    def get_tags(self):
        return []

    def get_tensors_to_dump(self):
        return {}

    # -------------------------- Local and Global training distinction
    def register_globally_trained_variables(self, l):
        if not self.train_only_globally:
            return
        for v in self.trainable_variables:
            l.append(v)

    def _set_regularization_loss(self):
        """
        L1 Regularization of the head variables
        """
        self.l1_regularization_loss_raw = None
        self.l1_regularization_loss = None
        if self.head_l1_regularization is None:
            return 0

        reg_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES,
            scope=self.scope,
        )
        self.l1_regularization_loss_raw = tf.reduce_sum(reg_losses)
        self.l1_regularization_loss = \
            self.head_l1_regularization * self.l1_regularization_loss_raw

    def _get_head_training_loss(self):
        if self.train_only_globally:
            return None

        if self.l1_regularization_loss is None:
            return self.loss
        else:
            return self.loss + self.l1_regularization_loss

    def _get_global_loss_contribution(self):
        if self.loss_weight_in_global_loss is None:
            return None

        loss_reg = self.l1_regularization_loss
        loss_weighted = self.loss * self.loss_weight_in_global_loss

        # Handle regularization if no head-only training is done
        if self.train_only_globally and loss_reg is not None:
            return loss_weighted + loss_reg
        return loss_weighted
