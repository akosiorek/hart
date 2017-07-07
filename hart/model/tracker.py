import numpy as np
import tensorflow as tf

from neurocity.component.model import Model
from neurocity.component import loss as _loss
from neurocity.tensor_ops import convert_shape, broadcast_against

from hart.model.attention_ops import AttentionCell, RATMAttention
from hart.train_tools import AdaptiveLoss
from hart.model.nn import MLP
from hart.model import tensor_ops


def ensure_array(value, n):
    value = np.asarray(value, dtype=np.float32)
    assert n % value.size == 0, 'Invalid shape: {} vs {}'.format(value.size, n)

    if value.size < n:
        value = np.tile(value, (n // value.size))
    return value


class HierarchicalAttentiveRecurrentTracker(Model):

    def __init__(self, inpt, bbox0, presence0, batch_size, glimpse_size,
                 feature_extractor, rnn_units, bbox_gain=-4., att_gain=-2.5,
                 zoneout_prob=0., identity_init=True, attention_module=RATMAttention, normalize_glimpse=False,
                 debug=False, clip_bbox=False, transform_init_features=False,
                 transform_init_state=False, dfn_readout=False, feature_shape=None, is_training=True):

        self.inpt = inpt
        self.bbox0 = bbox0
        self.presence0 = presence0
        self.glimpse_size = glimpse_size
        self.feature_extractor = feature_extractor
        self.rnn_units = rnn_units

        self.batch_size = batch_size
        self.inpt_size = convert_shape(inpt.get_shape()[2:], np.int32)
        self.bbox_gain = ensure_array(bbox_gain, 4)[np.newaxis]
        self.att_gain = ensure_array(att_gain, attention_module.n_params)[np.newaxis]
        self.zoneout_prob = zoneout_prob
        self.identity_init = identity_init
        self.attention_module = attention_module
        self.normalize_glimpse = normalize_glimpse
        self.debug = debug
        self.clip_bbox = clip_bbox
        self.transform_init_features = transform_init_features
        self.transform_init_state = transform_init_state
        self.dfn_readout = dfn_readout
        self.feature_shape = feature_shape
        self.is_training = tf.convert_to_tensor(is_training)

        super(HierarchicalAttentiveRecurrentTracker, self).__init__(self.__class__.__name__)
        try:
            self.register(is_training)
        except ValueError: pass

    def _build(self):
        self.cell = AttentionCell(self.feature_extractor,
                                  self.rnn_units, self.att_gain, self.glimpse_size, self.inpt_size,
                                  self.batch_size, self.zoneout_prob,
                                  self.attention_module, self.normalize_glimpse, self.identity_init,
                                  self.debug, self.dfn_readout, self.feature_shape, is_training=self.is_training)

        first_state = self.cell.zero_state(self.batch_size, tf.float32, self.bbox0, self.presence0, self.inpt[0],
                                           self.transform_init_features, self.transform_init_state)

        raw_outputs, state = tf.nn.dynamic_rnn(self.cell, self.inpt,
                                                        initial_state=first_state,
                                                        time_major=True,
                                                        scope=tf.get_variable_scope())

        if self.debug:
            (outputs, attention, presence, glimpse) = raw_outputs[:4]
            shape = (-1, self.batch_size, 1) + tuple(self.glimpse_size)
            self.glimpse = tf.reshape(glimpse, shape, 'glimpse_shape')
            tf.summary.histogram('rnn_outputs', outputs)
        else:
            (outputs, attention, presence) = raw_outputs[:3]

        if self.dfn_readout:
            self.obj_mask_logit = tf.reshape(raw_outputs[-3], (-1, self.batch_size, 1) + tuple(self.feature_shape))
            self.obj_mask = tf.nn.sigmoid(self.obj_mask_logit)
            obj_mask_features_flat = tf.reshape(raw_outputs[-2][1:], (-1, 10))
            self.dfn_weight_decay = raw_outputs[-1]

        self.rnn_output = outputs
        self.hidden_state = state[-1]
        self.raw_presence = presence
        self.presence = tf.nn.sigmoid(self.raw_presence)

        states_flat = tf.reshape(outputs[1:], (-1, self.rnn_units), 'flatten_states')
        if self.dfn_readout:
            states_flat = tf.concat(axis=1, values=(states_flat, obj_mask_features_flat))

        hidden_to_bbox = MLP(states_flat, self.rnn_units, 4, transfer=tf.nn.tanh, name='fc_h2bbox',
                             weight_init=self.cell._rec_init, bias_init=tf.constant_initializer())

        if self.debug:
            tf.summary.histogram('bbox_diff', hidden_to_bbox)

        attention = tf.reshape(attention, (-1, self.batch_size, 1, self.cell.att_size), 'shape_attention')
        self.attention = tf.concat(axis=0, values=(self.cell.att0[tf.newaxis], attention[:-1]))
        self.att_pred_bbox = self.cell.attention.attention_to_bbox(self.attention)
        self.att_pred_bbox_wo_bias = self.cell.attention.attention_to_bbox(self.attention - self.cell.att_bias)
        self.att_region = self.cell.attention.attention_region(self.attention)

        pred_bbox_delta = tf.reshape(hidden_to_bbox.output, (-1, self.batch_size, 1, 4), 'shape_pred_deltas')
        p = tf.zeros_like(pred_bbox_delta[0])[tf.newaxis]
        p = tf.concat(axis=0, values=(p, pred_bbox_delta))

        self.corr_pred_bbox = p * np.tile(self.inpt_size[:2], (2,)).reshape(1, 4)
        self.pred_bbox = self.att_pred_bbox_wo_bias + self.corr_pred_bbox

    def loss(self, target_bbox, target_presence, scale=None, loss_type='mean_squared_error',
             time_upscale=True, crossentropy_weight=.1, att_loss_type='mean_squared_error', att_weight=0.,
             obj_mask_weight=1., adaptive=False, l2_weight=0.):

        if not hasattr(self, '_loss'):
            losses = AdaptiveLoss()
            t0 = 1#0 if self.adjust_attention else 1
            float_presence = tf.to_float(target_presence)

            if loss_type.lower() == 'intersection_over_union':
                bbox_loss = _iou_loss(self.pred_bbox[t0:], target_bbox[t0:], float_presence[t0:])
            else:
                loss_func = getattr(_loss, loss_type)
                if not scale:
                    scale = lambda x: x

                yy, pred_y = (scale(i) for i in (target_bbox, self.pred_bbox))
                pred_delta = pred_y[1:] - pred_y[:-1]
                delta_y = yy[1:] - yy[:-1]

                delta_loss = loss_func(delta_y, pred_delta, target_presence[1:])

                len_weights = tf.reduce_sum(float_presence, axis=0, keep_dims=True)
                len_weights = 1. / tf.where(tf.greater(len_weights, 0), len_weights, tf.ones_like(len_weights))

                if time_upscale:
                    s = tf.to_float(tf.range(1, tf.shape(yy)[0]))[:, tf.newaxis, tf.newaxis]
                    len_weights = s * len_weights
                scaled_integrated_loss = loss_func(yy[t0:], pred_y[t0:], float_presence[t0:], len_weights)

                bbox_loss = 0.5 * delta_loss + 0.5 * scaled_integrated_loss

            losses.add('bbox', bbox_loss, adaptive=adaptive)
            if crossentropy_weight > 0.:
                xe = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.raw_presence, labels=float_presence))
                losses.add('xe', xe, crossentropy_weight, adaptive=adaptive)

            if att_weight > 0.:
                if att_loss_type == 'intersection':
                    att_loss = _intersection_loss(self.att_region[t0:], target_bbox[t0:], float_presence[t0:])
                elif att_loss_type == 'tight_intersection':
                    att_loss = _intersection_loss(self.att_region[t0:], target_bbox[t0:], float_presence[t0:])
                    att_loss += _area_loss(self.att_region[t0:], self.inpt_size[:2], float_presence[t0:])

                elif att_loss_type == 'intersection_over_union':
                    att_loss = _iou_loss(self.att_region[t0:], target_bbox[t0:], float_presence[t0:])
                else:
                    loss_func = getattr(_loss, att_loss_type)
                    att_size, bbox_size = self.att_region[t0:, ..., 2:], target_bbox[t0:, ..., 2:]

                    att_size = tf.where(tf.less(att_size, bbox_size), att_size, bbox_size)
                    size_penalty = loss_func(att_size, bbox_size, float_presence[t0:])

                    att_center = self.att_region[t0:, ..., :2] + .5 * att_size
                    bbox_center = target_bbox[t0:, ..., :2] + .5 * bbox_size

                    center_penalty = loss_func(att_center, bbox_center, float_presence[t0:])
                    att_loss = size_penalty + center_penalty

                losses.add('att', att_loss, att_weight, adaptive=adaptive)

            if self.dfn_readout and obj_mask_weight > 0.:
                # TODO: why does that work WITHOUT scale??!!

                # self.obj_mask_scale = (self.feature_shape / self.glimpse_size[:2].astype(np.float32))[np.newaxis, ...]
                target_mask_bbox = tensor_ops.intersection_within(target_bbox, self.att_pred_bbox)

                # self.target_obj_mask_bbox = np.tile(self.obj_mask_scale[np.newaxis], (1, 2)) * target_mask_bbox
                self.target_obj_mask_bbox = target_mask_bbox

                att_size = self.att_pred_bbox[..., 2:]
                self.target_obj_mask = tensor_ops.bbox_to_mask(self.target_obj_mask_bbox, att_size, self.feature_shape)

                obj_mask_xe = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.obj_mask_logit, labels=self.target_obj_mask)
                # obj_mask_xe = tf.reduce_mean(obj_mask_xe)

                # Handle pos/neg imbalance and use only present ones
                pos = tf.reduce_sum(self.target_obj_mask, (3, 4))
                neg = tf.reduce_sum(1 - self.target_obj_mask, (3, 4))
                frac_pos = pos / (pos + neg)
                frac_neg = 1 - frac_pos
                frac_pos, frac_neg = (broadcast_against(i, self.target_obj_mask) for i in (frac_pos, frac_neg))

                ones = tf.ones_like(self.target_obj_mask)
                gt_bool_mask = tf.cast(self.target_obj_mask, tf.bool)
                weight = tf.where(gt_bool_mask, ones * frac_pos, ones * frac_neg)
                weight = 1. / tf.where(tf.equal(weight, 0.), ones, weight)

                obj_mask_xe = _loss._apply_mask_and_weights(obj_mask_xe, target_presence, weight)
                losses.add('obj_mask', obj_mask_xe, obj_mask_weight, adaptive=adaptive)

                # iou
                pred_binary_mask = tf.round(self.obj_mask)
                pred_bool_mask = tf.cast(pred_binary_mask, bool)
                intersection = tf.reduce_sum(tf.to_float(tf.logical_and(pred_bool_mask, gt_bool_mask)), (3, 4))
                union = tf.reduce_sum(tf.to_float(tf.logical_or(pred_bool_mask, gt_bool_mask)), (3, 4))

                iou_mask = tf.logical_and(target_presence, tf.greater(union, 0.))
                obj_iou = _loss._mask_and_reduce(intersection / union, iou_mask)
                losses.track('obj_iou', obj_iou)

                # accuracy
                acc = tf.to_float(tf.equal(pred_binary_mask, self.target_obj_mask))
                acc = _loss._mask_and_reduce(acc, target_presence)
                losses.track('obj_acc', acc)

            if l2_weight > 0.:
                weights = [v for v in tf.trainable_variables() if 'W_' in v.name or '_W' in v.name]
                l2_loss = sum((tf.nn.l2_loss(w) for w in weights))

                if self.dfn_readout:
                    l2_loss += tf.reduce_mean(self.dfn_weight_decay)

                losses.add('l2', l2_weight * l2_loss, 1., adaptive=False)

            losses.track('iou', self.iou(target_bbox, target_presence))
            self._loss = losses

        return self._loss

    def iou(self, target_bbox, presence, per_timestep=False, reduce=True, start_t=1):

        pred_bbox, target_bbox, presence = [i[start_t:] for i in (self.pred_bbox, target_bbox, presence)]
        if not per_timestep:
            return _loss.intersection_over_union(pred_bbox, target_bbox, presence)
        else:
            iou = _loss.intersection_over_union(pred_bbox, target_bbox, reduce=False)
            iou = tf.where(presence, iou, tf.zeros_like(iou))
            iou = tf.reduce_sum(iou, (1, 2))
            p = tf.reduce_sum(tf.to_float(presence), (1, 2))
            if reduce:
                p = tf.maximum(p, tf.ones(tf.shape(presence)[0]))
                iou /= p
                return iou
            else:
                return iou, p


def _mask(expr, mask):
    mask = tf.cast(broadcast_against(mask, expr), expr.dtype)
    mask = tf.ones(tf.shape(expr), dtype=mask.dtype) * mask
    return tf.where(tf.cast(mask, bool), expr, tf.zeros_like(expr))


def _time_weighted_nll(expr, presence, weights=None):
    nll = _loss.negative_log_likelihood(expr, reduce=False, weights=weights)
    nll = _mask(nll, presence)
    p = tf.reduce_sum(presence, 1)
    nll = tf.reduce_sum(nll, 1) / p
    return _loss._mask_and_reduce(nll, mask=tf.greater(p, 0))


def _iou_loss(pred_bbox, target_bbox, presence):
    i = _loss.intersection_over_union(pred_bbox, target_bbox, reduce=False)
    # return _loss.negative_log_likelihood(i, presence)
    # weight by the number of instances present at time t
    return _time_weighted_nll(i, presence)


def _intersection_loss(pred_bbox, target_bbox, presence):
    area = target_bbox[..., 2] * target_bbox[..., 3]
    i = _loss.intersection(pred_bbox, target_bbox, reduce=False)
    i /= tf.where(tf.greater(area, 0), area, tf.ones_like(i))
    return _time_weighted_nll(i, presence)


def _area_loss(pred_bbox, img_size, presence):
    area = pred_bbox[..., 2] * pred_bbox[..., 3]
    ratio = area / tf.reduce_prod(tf.to_float(img_size))
    weights = tf.clip_by_value(ratio, 1., 10.)
    ratio = tf.clip_by_value(ratio, 0., 1.)
    return _time_weighted_nll(1 - ratio, presence, weights)

