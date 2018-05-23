import tensorflow as tf
import numpy as np
from base import NetworkHeadBase
import src.features as ft_def
from src.features import all_features


class LoopBufferManager:
    def __init__(self, size, default_value=0.0, name='loop_buffer'):
        with tf.name_scope(name):
            self.cursor_pos = tf.Variable(
                0,
                trainable=False,
                name='cursor_pos',
            )
            self.total_written = tf.Variable(
                0,
                trainable=False,
                name='total_written',
            )
            self.buffer = tf.Variable(
                [default_value] * size,
                name='buffer',
                trainable=False,
            )
        self.buffer_size = size
        self.name = name

    def insert_values(self, values):
        values = tf.reshape(values, [-1])
        num_values = tf.shape(values)[0]
        write_pos_last = tf.minimum(
            self.cursor_pos + num_values,
            self.buffer_size,
        )

        buf = self.buffer.assign(
            tf.concat([
                self.buffer[:self.cursor_pos],
                values[:write_pos_last - self.cursor_pos],
                self.buffer[write_pos_last:],
            ], 0),
        )
        with tf.control_dependencies([
            buf,
            self.total_written.assign_add(num_values),
        ]):
            return self.cursor_pos.assign(tf.cond(
                self.cursor_pos + num_values >= self.buffer_size,
                lambda: 0,
                lambda: self.cursor_pos + num_values,
            ))

    def get_buffer(self):
        return self.buffer[0:self.total_written]


def accumulated_histogram(variable, accumulate_size, name):
    buf = LoopBufferManager(
        accumulate_size,
        name='%s/buf' % name,
    )
    update_op = buf.insert_values(variable)
    summary = tf.summary.histogram(
        name,
        buf.get_buffer(),
        collections=[],
    )
    return summary, update_op


class ClassificationHead(NetworkHeadBase):
    def __init__(
        self,
        # NetworkHeadBase arguments
        name,
        model,
        last_layer,
        features,
        # Custom arguments (from config file)
        predict,
        loss_classes_weights={},
        num_classes_in_evaluation=None,
        # Args passed to parent
        **kwargs
    ):
        """
        Arguments:
        - predict:
          List of boolean features to predict
          Example: ['is_class_a', 'is_class_b']
        """
        if num_classes_in_evaluation is None:
            num_classes_in_evaluation = len(predict)
        assert(num_classes_in_evaluation <= len(predict))
        self.model = model
        self.num_classes_in_evaluation = num_classes_in_evaluation
        self.predict_feature_names = predict
        self.features = features
        self.metrics = {}

        # Compute loss
        weights_per_class = {ft_name: 1.0 for ft_name in predict}
        weights_per_class.update(loss_classes_weights)
        weights_per_class = np.array([
            weights_per_class[ft_name]
            for ft_name in predict
        ])
        self.labels = [features[ft_name] for ft_name in predict]
        self.labels = tf.concat(self.labels, 1)
        self.predictions, self.loss, weighted_loss_per_sample = \
            self.compute_prediction_and_loss(
                last_layer,
                weights_per_class,
            )
        self.predictions = tf.identity(self.predictions, name='logits')

        with tf.variable_scope('metrics'):
            self.compute_metrics(weighted_loss_per_sample)

        super(ClassificationHead, self).__init__(
            name=name,
            model=model,
            last_layer=last_layer,
            features=features,
            **kwargs
        )

    def compute_metrics(self, weighted_loss_per_sample):
        # Metrics for training/eval
        batch_size = tf.cast(tf.shape(self.labels)[0], tf.float32)
        accuracy = tf.cast(
            tf.equal(
                tf.argmax(self.predictions, 1),
                tf.argmax(self.labels, 1),
            ),
            tf.float32,
        )
        self.metrics.update({
            'accuracy': tf.reduce_mean(accuracy),
        })
        self.metrics.update({
            'labels/%s_ratio' % class_name: tf.reduce_mean(
                tf.cast(self.labels[:, i], tf.float32),
            )
            for i, class_name in enumerate(self.predict_feature_names)
        })
        self.metrics.update({
            'logits/%s' % class_name: tf.reduce_mean(
                tf.cast(self.predictions[:, i], tf.float32),
            )
            for i, class_name in enumerate(self.predict_feature_names)
        })
        self.metrics.update({
            'predicted_%s_ratio' % ft_name:
            tf.reduce_sum(tf.cast(
                tf.equal(tf.argmax(self.predictions, 1), i),
                tf.float32,
            )) / batch_size
            for i, ft_name in enumerate(self.predict_feature_names)
        })

        # Accuracy on different classes
        def accuracy_on_class(i):
            weights = tf.cast(
                tf.equal(
                    tf.argmax(self.labels, 1),
                    i,
                ),
                tf.float32,
            )
            class_acc = tf.reduce_sum(weights * accuracy)
            class_acc /= tf.reduce_sum(weights) + 0.0001
            return class_acc, tf.reduce_sum(weights)
        self.classes_accuracy = {
            'accuracy_on_%s' % c: accuracy_on_class(i)
            for i, c in enumerate(self.predict_feature_names)
        }
        self.metrics.update({
            name: v[0]
            for name, v in self.classes_accuracy.items()
        })
        if weighted_loss_per_sample is not None:
            labels_float = tf.cast(self.labels, tf.float32)
            loss_contribution_ratio = [
                tf.reduce_sum(
                    weighted_loss_per_sample * labels_float[:, i] / batch_size
                )
                for i, _ in enumerate(self.predict_feature_names)
            ]
            self.metrics.update({
                'loss_contribution_ratio/%s' % class_name:
                    loss_contribution_ratio[i] / self.loss
                for i, class_name in enumerate(self.predict_feature_names)
            })

        # Histogram summaries
        self.probas = tf.nn.softmax(self.predictions)
        control_deps = []
        if self.model.is_training_bool:
            HISTOGRAM_NUMBER_VALUES_TRAIN_SET = 350
            for p_idx, p_ft_name in enumerate(self.predict_feature_names):
                summary, update_op = accumulated_histogram(
                    self.probas[:, p_idx],
                    HISTOGRAM_NUMBER_VALUES_TRAIN_SET,
                    'p_%s' % p_ft_name,
                )
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, summary)
                control_deps.append(update_op)
                summary, update_op = accumulated_histogram(
                    self.predictions[:, p_idx],
                    HISTOGRAM_NUMBER_VALUES_TRAIN_SET,
                    'logits_%s' % p_ft_name,
                )
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, summary)
                control_deps.append(update_op)
        # Loss with all required operations
        with tf.control_dependencies(control_deps):
            self.loss = tf.identity(self.loss)

    def compute_prediction_and_loss(
        self,
        last_layer,
        weights_per_class,
    ):
        with tf.variable_scope('prediction'):
            predictions = self.model.gen_head(
                last_layer,
                len(self.predict_feature_names),
            )
        with tf.variable_scope('loss'):
            weighted_loss_per_sample = tf.losses.softmax_cross_entropy(
                self.labels,
                predictions,
                reduction=tf.losses.Reduction.NONE,
                weights=tf.reduce_sum(
                    tf.cast(self.labels, tf.float32) * weights_per_class, 1,
                ),
            )
            loss = tf.reduce_mean(weighted_loss_per_sample)
            return predictions, loss, weighted_loss_per_sample

    def _get_evaluated_metrics(self):
        evaluation_metrics = \
            super(ClassificationHead, self)._get_evaluated_metrics()

        predicted = tf.argmax(
            self.predictions[:, 0:self.num_classes_in_evaluation],
            1,
        )
        actual_class = tf.argmax(self.labels, 1)
        evaluation_metrics.update({
            'false_negatives': tf.metrics.false_negatives(
                actual_class, predicted),
            'false_positives': tf.metrics.false_positives(
                actual_class, predicted),
            'mean_per_class_accuracy': tf.metrics.mean_per_class_accuracy(
                actual_class, predicted, self.num_classes_in_evaluation),
        })
        evaluation_metrics.update({
            n: tf.metrics.mean(v[0], weights=v[1], name='%s_weighted' % n)
            for n, v in self.classes_accuracy.items()
        })

        # Histograms
        HISTOGRAM_NUMBER_VALUES_TEST_SET = 330
        for p_idx, p_ft_name in enumerate(self.predict_feature_names):
            summary, update_op = accumulated_histogram(
                self.probas[:, p_idx],
                HISTOGRAM_NUMBER_VALUES_TEST_SET,
                'p_%s' % p_ft_name,
            )
            evaluation_metrics.update({
                'p_%s' % p_ft_name: (summary, update_op)
            })
            summary, update_op = accumulated_histogram(
                self.predictions[:, p_idx],
                HISTOGRAM_NUMBER_VALUES_TEST_SET,
                'logits_%s' % p_ft_name,
            )
            evaluation_metrics.update({
                'logits_%s' % p_ft_name: (summary, update_op)
            })
        return evaluation_metrics

    def get_predictions(self):
        predictions = super(ClassificationHead, self).get_predictions()
        predictions.update({
            ft_name: self.predictions[:, i]
            for i, ft_name in enumerate(self.predict_feature_names)
        })
        return predictions

    def get_tags(self):
        tags = super(ClassificationHead, self).get_tags()
        tags.append('classification')
        feature_info = all_features.feature_info
        ft_names = [
            feature_info[n]['shortname']
            for n in self.predict_feature_names
        ]
        ft_names.sort()
        tags.append('_'.join(ft_names))
        return tags

    def get_tensors_to_dump(self):
        if self.model.is_training_bool:
            return {}
        return {
            'proba': self.probas,
            'labels': self.labels,
            'logits': self.predictions,
            'image_label': self.features[ft_def.IMAGE_LABEL],
            'subject_label': self.features[ft_def.SUBJECT_LABEL],
            'dataset': self.features[ft_def.DATASET],
            'study_patient_id': self.features[ft_def.STUDY_PATIENT_ID],
            'study_image_id': self.features[ft_def.STUDY_IMAGE_ID],
        }


class ClassificationSVMHead(ClassificationHead):
    def __init__(
        self,
        svm_reg_coeff=5.0,
        svm_force_w_symmetry_axis=[],
        **kwargs
    ):
        self.svm_reg_coeff = svm_reg_coeff
        self.svm_force_w_symmetry_axis = svm_force_w_symmetry_axis
        super(ClassificationSVMHead, self).__init__(**kwargs)

    """
    Does one-versus-rest classification using SVM loss
    """
    def compute_prediction_and_loss(
        self,
        last_layer,
        weights_per_class,
    ):
        SVM_REG_CONST = self.svm_reg_coeff

        all_predictions = []
        all_loss = tf.constant(0.0)
        for i, ft_name in enumerate(self.predict_feature_names):
            ft_shortname = all_features.feature_info[ft_name]['shortname']
            with tf.variable_scope('%s_vs_rest' % (ft_shortname)):
                # Prediction and loss
                predictions, W = self.build_svm_layer(last_layer)
                weighted_loss_per_sample = tf.losses.hinge_loss(
                    tf.cast(self.labels[:, i], tf.float32),
                    tf.reshape(predictions, [-1]),
                    reduction=tf.losses.Reduction.NONE,
                    weights=tf.reduce_sum(tf.cast(
                        self.labels, tf.float32) * weights_per_class, 1,
                    ),
                )
                w_l2 = 0.5 * tf.reduce_sum(tf.square(W))
                reg_loss = SVM_REG_CONST * w_l2
                hinge_loss = tf.reduce_mean(weighted_loss_per_sample)
                loss = hinge_loss + reg_loss
                self.metrics.update({
                    '%s_vs_rest/hinge_loss' % ft_shortname: hinge_loss,
                    '%s_vs_rest/w_l2' % ft_shortname: w_l2,
                    '%s_vs_rest/reg_loss' % ft_shortname: reg_loss,
                    '%s_vs_rest/reg_loss_ratio' % ft_shortname:
                        reg_loss / loss,
                })
                all_predictions.append(predictions)
                all_loss += loss
        return tf.concat(all_predictions, 1), all_loss, None

    def build_svm_layer(self, last_layer):
        in_shape = last_layer.get_shape().as_list()[1:]
        W = tf.Variable(tf.zeros(in_shape), name="svm_w")
        b = tf.Variable(tf.zeros([1]), name="svm_b")
        if len(self.svm_force_w_symmetry_axis) > 0:
            W = 0.5 * (W + tf.reverse(W, self.svm_force_w_symmetry_axis))
        reduction_axis = range(1, len(in_shape) + 1)
        y_raw = tf.reduce_sum(last_layer * W, axis=reduction_axis) + b
        return tf.reshape(y_raw, [-1, 1]), W

    def get_tags(self):
        tags = super(ClassificationSVMHead, self).get_tags()
        tags.append('svm_head')
        return tags
