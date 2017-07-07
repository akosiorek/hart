import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMCell
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.util import nest

from hart.model import tensor_ops
from hart.model.nn import DynamicFilterModel
from hart.model.rnn import ZoneoutWrapper, IdentityLSTMCell
from neurocity.component.layer import AffineLayer, ConvLayer
from neurocity.tensor_ops import convert_shape


def gaussian_mask(params, R, C):
    """Define a mask of size RxC given by one 1-D Gaussian per row.

    u, s and d must be 1-dimensional vectors"""
    u, s, d = (params[..., i] for i in xrange(3))

    for i in (u, s, d):
        assert len(u.get_shape()) == 1, i

    batch_size = tf.to_int32(tf.shape(u)[0])

    R = tf.range(tf.to_int32(R))
    C = tf.range(tf.to_int32(C))
    R = tf.to_float(R)[tf.newaxis, tf.newaxis, :]
    C = tf.to_float(C)[tf.newaxis, :, tf.newaxis]
    C = tf.tile(C, (batch_size, 1, 1))

    u, d = u[:, tf.newaxis, tf.newaxis], d[:, tf.newaxis, tf.newaxis]
    s = s[:, tf.newaxis, tf.newaxis]

    ur = u + (R - 0.) * d
    sr = tf.ones_like(ur) * s

    mask = C - ur
    mask = tf.exp(-.5 * (mask / sr) ** 2)

    mask /= tf.reduce_sum(mask, 1, keep_dims=True) + 1e-8
    return mask


def extract_glimpse(inpt, attention_params, glimpse_size):
    """Extracts an attention glimpse

    :param inpt: tensor of shape == (batch_size, img_height, img_width)
    :param attention_params: tensor of shape = (batch_size, 6) as
        [uy, sy, dy, ux, sx, dx] with u - mean, s - std, d - stride"
    :param glimpse_size: 2-tuple of ints as (height, width),
        size of the extracted glimpse
    :return: tensor
    """

    ap = attention_params
    shape = inpt.get_shape()
    rank = len(shape)

    assert rank in (3, 4), "Input must be 3 or 4 dimensional tensor"

    inpt_H, inpt_W = shape[1:3]
    if rank == 3:
        inpt = inpt[..., tf.newaxis]
        rank += 1

    Fy = gaussian_mask(ap[..., 0::2], glimpse_size[0], inpt_H)
    Fx = gaussian_mask(ap[..., 1::2], glimpse_size[1], inpt_W)

    gs = []
    for channel in tf.unstack(inpt, axis=rank - 1):
        g = tf.matmul(tf.matmul(Fy, channel, adjoint_a=True), Fx)
        gs.append(g)
    g = tf.stack(gs, axis=rank - 1)

    g.set_shape([shape[0]] + list(glimpse_size))
    return g


class Attention(object):
    n_params = None

    def __init__(self, inpt_size, glimpse_size):

        self.inpt_size = np.asarray(inpt_size)
        self.glimpse_size = np.asarray(glimpse_size)

    def extract_glimpse(self, inpt, raw_att, return_all=False):
        raw_att_flat = tf.reshape(raw_att, (-1, self.n_params), 'flat_raw_att')
        att_flat = self._to_attention(raw_att_flat)

        shape = raw_att.get_shape().as_list()
        n_glimpses = int(raw_att.get_shape()[-1]) // self.n_params

        att = tf.reshape(att_flat, shape[:-1] + [n_glimpses, int(att_flat.get_shape()[-1])])
        glimpse = []
        for a in tf.unstack(att, axis=1):
            glimpse.append(self._extract_glimpse(inpt, a))

        glimpse = tf.stack(glimpse, 1)
        glimpse = tf.reshape(glimpse, (-1,) + tuple(self.glimpse_size))

        if return_all:
            return raw_att_flat, att_flat, glimpse
        else:
            return glimpse

    def attention_to_bbox(self, att):
        with tf.variable_scope('attention_to_bbox'):
            yx = att[..., :2] * self.inpt_size[np.newaxis, :2]
            hw = att[..., 2:4] * (self.inpt_size[np.newaxis, :2] - 1)
            bbox = tf.concat(axis=tf.rank(att) - 1, values=(yx, hw))
            bbox.set_shape(att.get_shape()[:-1].concatenate((4,)))
        return bbox

    def attention_region(self, att):
        return self.attention_to_bbox(att)

    def _extract_glimpse(self, inpt, att_flat):
        return extract_glimpse(inpt, att_flat, self.glimpse_size)


class RATMAttention(Attention):
    """Implemented after https://arxiv.org/abs/1510.08660"""
    n_params = 6

    def bbox_to_attention(self, bbox):
        with tf.variable_scope('ratm_bbox_to_attention'):
            us = bbox[..., :2] / self.inpt_size[np.newaxis, :2]
            ss = 0.5 * bbox[..., 2:] / self.inpt_size[np.newaxis, :2]
            ds = bbox[..., 2:] / (self.inpt_size[np.newaxis, :2] - 1.)

            att = tf.concat(axis=tf.rank(bbox) - 1, values=(us, ss, ds))
        return att

    @staticmethod
    def _to_axis_attention(params, glimpse_dim, inpt_dim):
        u, s, d = (params[..., i] for i in xrange(RATMAttention.n_params // 2))
        u = u * inpt_dim
        s = (s + 1e-5) * float(inpt_dim) / glimpse_dim
        d = d * float(inpt_dim - 1) / (glimpse_dim - 1)
        return u, s, d

    def _to_attention(self, params):
        (y, x), (u, v) = self.inpt_size[:2], self.glimpse_size[:2]
        uy, sy, dy = self._to_axis_attention(params[..., ::2], u, y)
        ux, sx, dx = self._to_axis_attention(params[..., 1::2], v, x)

        ap = (uy, ux, sy, sx, dy, dx)
        ap = tf.transpose(tf.stack(ap), name='attention')
        assert ap.get_shape()[-1] == self.n_params, 'Invalid attention shape={}!'.format(ap.get_shape())
        return ap


class FixedStdAttention(Attention):
    """Like RATM but std for the gaussian mask depends directly and exclusively on the stride
    between gaussians. I used a small neural net to compute std to approximate bicubic
    interpolation and then fitted a 4th order surface to predictions of the neural net.

    There's also an additive (learnt) bias in pixels to the upper left corner of the attention
     window."""
    n_params = 4
    offset_bias = np.asarray([0.00809737, 0.50086582], dtype=np.float32).reshape(1, 2)
    weights = np.asarray([[6.12598441e-01, 9.25613308e-01],
                          [-1.05801568e-02, -2.18224973e-03],
                          [1.32131897e-04, -6.09307166e-06],
                          [-2.87635530e-07, 9.08051012e-08],
                          [1.94529164e-10, -9.47235313e-11],
                          [1.44468477e-04, -1.19733592e-02],
                          [-4.30590720e-06, 7.71485474e-05],
                          [1.05376852e-08, -1.05474865e-07],
                          [-6.49625282e-12, 3.43567810e-11],
                          [9.85685680e-06, 6.57580098e-05],
                          [-1.41991381e-08, -9.46024867e-08],
                          [-9.81123812e-09, -1.56167932e-07],
                          [-3.61024557e-12, 7.68954027e-11],
                          [2.65848501e-11, 4.39530612e-11],
                          [-1.01850187e-11, 9.85289183e-11]], dtype=np.float32)

    def bbox_to_attention(self, bbox):
        with tf.variable_scope('fixed_std_bbox_to_attention'):
            us = bbox[..., :2] / self.inpt_size[np.newaxis, :2]
            ds = bbox[..., 2:] / (self.inpt_size[np.newaxis, :2] - 1.)

            att = tf.concat(axis=tf.rank(bbox) - 1, values=(us, ds))
            att.set_shape(bbox.get_shape()[:-1].concatenate([4]))
        return att

    def _stride_to_std(self, stride):
        shape = convert_shape(stride.get_shape())
        stride_flat = tf.reshape(stride, (-1, shape[-1]))
        y, x = stride_flat[..., 0], stride_flat[..., 1]
        features = [
            tf.ones_like(y),
            y, y ** 2, y ** 3, y ** 4,
            x, x ** 2, x ** 3, x ** 4,
               y * x, y * x ** 2, y ** 2 * x,
               y * x ** 3, y ** 2 * x ** 2, y ** 3 * x
        ]

        features = tf.concat(axis=1, values=[f[..., tf.newaxis] for f in features])
        sigma_flat = tf.matmul(features, self.weights)
        return tf.reshape(sigma_flat, shape)

    def _to_attention(self, raw_att, with_bias=True):
        bbox = FixedStdAttention.attention_to_bbox(self, raw_att)
        us = bbox[..., :2]
        if with_bias:
            us += self.offset_bias

        ds = bbox[..., 2:4] / (self.glimpse_size[np.newaxis, :2] - 1)
        ss = self._stride_to_std(ds)

        ap = tf.concat(axis=tf.rank(raw_att) - 1, values=(us, ss, ds), name='attention')
        ap.set_shape(raw_att.get_shape()[:-1].concatenate((6,)))
        return ap


class AttentionCell(RNNCell):
    def __init__(self, feature_extractor, n_units, att_gain, glimpse_size,
                 input_size=None, batch_size=None,
                 zoneout_prob=0., attention_module=RATMAttention,
                 normalize_glimpse=False, identity_init=True, debug=False,
                 predict_appearance=False, feature_shape=None, is_training=True):

        assert len(glimpse_size) in (2, 3), 'Invalid size'
        assert input_size is None or len(input_size) == len(glimpse_size), 'Invalid size'

        self.feature_extractor = feature_extractor
        self.n_units = n_units
        self.att_gain = att_gain
        self.glimpse_size = glimpse_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.normalize_glimpse = normalize_glimpse
        self.identity_init = identity_init
        self.debug = debug
        self.predict_appearance = predict_appearance

        self.attention = attention_module(self.input_size, self.glimpse_size)

        if not isinstance(zoneout_prob, (tuple, list)):
            zoneout_prob = (zoneout_prob, 0.)

        self.zoneout_prob = zoneout_prob

        self.cell = self._make_cell(is_training)
        self._rec_init = tf.random_uniform_initializer(-1e-3, 1e-3)

        self._att_size = self.att_size
        self._state_size = (self._att_size, 1, self.cell.state_size)
        self._output_size = (self.cell.output_size, self._att_size, 1)

        if self.debug:
            self._output_size += (np.prod(self.glimpse_size),)

        if self.predict_appearance:
            self._state_size += (self.n_units,)
            self._output_size += (np.prod(feature_shape), 10, 1)

    @property
    def att_size(self):
        return self.attention.n_params

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def zero_state(self, batch_size, dtype, bbox0=None, presence=None, img0=None,
                   transform_featuers=False, transform_state=False):
        if bbox0 is None:
            return super(AttentionCell, self).zero_state(batch_size, dtype)

        self.att0 = self.attention.bbox_to_attention(bbox0)
        att0 = tf.reshape(self.att0, (-1, self.att_size), 'shape_first_att')
        presence = tf.to_float(tf.reshape(presence, (-1, 1)), 'shape_first_presence') * 1e3

        state = att0, presence
        zero_state = self.cell.zero_state(batch_size, dtype)
        if img0 is not None:
            zero_state, rnn_outputs = self._zero_state(img0, att0, presence, zero_state,
                                                       transform_featuers, transform_state)

            att_bias = tf.get_variable('att_bias', (1, 1, 1, self.att_size), initializer=self._rec_init)
            self.att_bias = .25 * tf.nn.tanh(att_bias)
            att0 += tf.reshape(tf.tile(self.att_bias, (1, 1, 1, 1)), (1, -1))
            self.att0 += self.att_bias[0]
            state += (zero_state,)
            if self.predict_appearance:
                rnn_outputs = tf.reshape(rnn_outputs, (-1, self.n_units))
                state += (rnn_outputs,)
        else:
            state += (zero_state,)

        return state

    def _zero_state(self, img, att, presence, state, transform_features, transform_state=False):

        with tf.variable_scope(self.__class__.__name__) as vs:
            features = self.extract_features(img, att)[1]

            if transform_features:
                features_flat = tf.reshape(features, (-1, self.n_units))
                features_flat = AffineLayer(features_flat, self.n_units, name='init_feature_transform').output
                features = tf.reshape(features_flat, tf.shape(features))

            rnn_outputs, hidden_state = self._propagate(features, state)

            hidden_state = nest.flatten(hidden_state)

            if transform_state:
                for i, hs in enumerate(hidden_state):
                    name = 'init_state_transform_{}'.format(i)
                    hidden_state[i] = AffineLayer(hs, self.n_units, name=name).output

            state = nest.pack_sequence_as(structure=state, flat_sequence=hidden_state)
        self.rnn_vs = vs
        return state, rnn_outputs

    def _make_cell(self, is_training):

        raw_cell = IdentityLSTMCell if self.identity_init else LSTMCell

        if self.zoneout_prob[0] > 0.:
            cell = lambda: ZoneoutWrapper(raw_cell(self.n_units), self.zoneout_prob, is_training)
        else:
            cell = lambda: raw_cell(self.n_units)

        return cell()

    def _propagate(self, inpt, state):
        features = tf.reshape(inpt, (self.batch_size, self.n_units))
        outputs, hidden_state = self.cell(features, state)
        return tf.reshape(outputs, (self.batch_size, 1, self.n_units)), hidden_state

    def extract_features(self, inpt, raw_att, apperance_vec=None, reuse=False):

        raw_att_flat, att_flat, glimpse_flat = self.attention.extract_glimpse(inpt, raw_att,
                                                                              return_all=True)
        if self.normalize_glimpse:
            # do not normalize depth
            colour = tensor_ops.normalize_contrast(glimpse_flat[..., :3])

            if glimpse_flat.get_shape()[-1] == 4:
                ax = len(glimpse_flat.get_shape()) - 1
                glimpse_flat = tf.concat(axis=ax, values=(colour, glimpse_flat[..., 3:]))
            else:
                glimpse_flat = colour

        features = self.feature_extractor(glimpse_flat, reuse=reuse)

        def flatten_features(f, name='', more_feats=None):
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                f_flat = tf.reshape(f, (self.batch_size * 1, -1), 'reshape' + name)
                to_concat = (f_flat, att_flat)
                if more_feats is not None:
                    to_concat += more_feats

                f_flat = tf.concat(axis=1, values=to_concat, name='concat_att' + name)
                return AffineLayer(f_flat, self.n_units, transfer=tf.nn.elu, name='fc_after_conv' + name)

        if apperance_vec is not None:
            before_features = self.feature_extractor.orig_output
            self.dfn_inpt = DynamicFilterModel(before_features, apperance_vec, ksize=(1, 1), n_channels=5,
                                               name='pre_DFN')
            self.dfn = DynamicFilterModel(self.dfn_inpt.features, apperance_vec, ksize=(3, 3), n_channels=5)
            dfn = self.dfn.features

            obj_mask_logit = ConvLayer(dfn, (1, 1), 1, transfer=None, name='obj_mask').output
            obj_mask = tf.nn.sigmoid(obj_mask_logit)
            features *= obj_mask

            flat_mask = tf.reshape(obj_mask, (self.batch_size * 1, -1))

        features_flat = flatten_features(features)

        features = tf.reshape(features_flat, (self.batch_size, 1, self.n_units), 'to_features')

        output = raw_att_flat, features, glimpse_flat
        if apperance_vec is not None:
            output += (obj_mask_logit, flat_mask)
        return output

    def __call__(self, inpt, state, scope=None):
        raw_att, presence, hidden_state = state[:3]
        if self.predict_appearance:
            apperance_vec = tf.reshape(state[3], (self.batch_size * 1, self.n_units))

        if self.batch_size is None:
            self.batch_size = int(inpt.get_shape()[0])

        if self.input_size is None:
            self.input_size = [tf.to_float(i) for i in inpt.get_shape()[-2:]]

        with tf.variable_scope(self.__class__.__name__):

            all_features = self.extract_features(inpt, raw_att, apperance_vec=apperance_vec, reuse=True)
            raw_att_flat, features, glimpses = all_features[:3]

            with tf.variable_scope(self.rnn_vs, initializer=self._rec_init, reuse=True):
                rnn_outputs, hidden_state = self._propagate(features, hidden_state)

            # delta-update of the raw attention params
            zero_init = tf.constant_initializer()
            outputs_flat = tf.reshape(rnn_outputs, (-1, self.n_units), 'outputs_flat')

            att_inpt = outputs_flat
            if self.predict_appearance:
                flat_mask = all_features[-1]
                mask_features = AffineLayer(flat_mask, 10, transfer=tf.nn.elu, name='mask_features')
                att_inpt = tf.concat(axis=1, values=(att_inpt, mask_features))

            att_readout = AffineLayer(att_inpt, self.n_units, transfer=tf.nn.elu, name='att_readout_1')
            att_diff_flat = AffineLayer(att_readout, self.att_size,
                                        transfer=tf.nn.tanh,
                                        weight_init=self._rec_init,
                                        bias_init=zero_init,
                                        name='att_readout')

            att_delta_scale = tf.Variable(self.att_gain, name='att_delta_scale')
            new_att_flat = raw_att_flat + tf.nn.sigmoid(att_delta_scale) * att_diff_flat.output

            new_att = tf.reshape(new_att_flat, (-1, self.att_size), 'new_att_shape')
            rnn_outputs = tf.reshape(rnn_outputs, (-1, self.n_units), 'outputs_shape')

        outputs, state = (rnn_outputs, new_att, presence), (new_att, presence, hidden_state)

        if self.debug:
            glimpse_flat = tf.reshape(glimpses, (self.batch_size, -1))
            outputs += (glimpse_flat,)

        if self.predict_appearance:
            rnn_outputs = tf.reshape(rnn_outputs, (self.batch_size, self.n_units))
            state += (rnn_outputs,)

            # concat flat obj_mask to outputs
            flat_obj_mask = tf.reshape(all_features[-2], (self.batch_size, -1))
            flat_mask_features = tf.reshape(mask_features, (self.batch_size, -1))

            # weight decay
            def weight_decay(w):
                l = tf.reshape(w, (self.batch_size, 1, -1))
                return tf.reduce_sum(l ** 2, (1, 2))[..., tf.newaxis] / (2 * 1)

            dynamic_weights = (
            self.dfn.dynamic_weights, self.dfn.dynamic_bias, self.dfn_inpt.dynamic_weights, self.dfn_inpt.dynamic_bias)
            dfn_weight_decay = sum((weight_decay(i) for i in dynamic_weights))

            outputs += (flat_obj_mask, flat_mask_features, dfn_weight_decay)

        return outputs, state
