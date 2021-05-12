#%%
import paddle as mi
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import logging

# homebrew
import misc
logger = logging.getLogger(__name__)

def calc_padding(kernel_size, stride=1, dilation=1):
    padding = ((dilation * (kernel_size - 1) + 1) - stride) / 2
    if padding != int(padding):
        logger.critical('Padding is NOT an integer!')
    else:
        padding = int(padding)
    return padding


def position_encoding_trig(x_shape, curve='trig'):
    """ x dim: [i=batch_size, j=seq_len, k=feature_dim] """
    # x = mi.empty((args.batch_size, 50, 128), # args.feature_dim), dtype='float32')
    assert len(x_shape) >= 2

    jlen = x_shape[-2]
    klen = x_shape[-1]

    j = np.linspace(1, jlen, num=jlen, dtype='float32').reshape((jlen,1))
    k = np.linspace(1, klen, num=klen, dtype='float32').reshape((1,klen))

    k = (k + 1) // 2
    omega_k = 1 / 10000**(2*k/klen)
    omega_jk = np.matmul(j, omega_k)

    omega_jk[0::2,:] = np.sin(omega_jk[0::2,:])
    omega_jk[1::2,:] = np.cos(omega_jk[1::2,:])

    return mi.to_tensor(omega_jk)


def get_attn_mask(x, seqs_len):
    """  """
    if seqs_len is None:
        return None
        
    batch_size, src_len, d_model = x.shape

    if all(seqs_len.numpy() == src_len):
        return None

    # mask is added to the product before softmax
    attn_mask = mi.full((batch_size, 1, src_len, src_len), 0) # -np.inf)

    for i in range(batch_size):
        # attn_mask[i, 0, :seqs_len[i], :seqs_len[i]] = 0
        attn_mask[i, 0, seqs_len[i]:, :seqs_len[i]] = -np.inf
        attn_mask[i, 0, :seqs_len[i], seqs_len[i]:] = -np.inf

    return attn_mask


class PositionEncoder(nn.Layer):
    """ better create a buffer to  """
    def __init__(self, input_size, curve='trig'):
        super(PositionEncoder, self).__init__()

        pos_mat = position_encoding_trig(input_size, curve=curve)
        self.register_buffer('pos_mat', pos_mat, persistable=False)

    def forward(self, x, beta=1.0):
        jlen = x.shape[-2]
        klen = x.shape[-1]
        return x + beta * self.pos_mat[:jlen, :klen]


class AttentionMask(nn.Layer):
    """ not a good idea """
    def __init__(self, max_len=1024):
        super(AttentionMask, self).__init__()

        attn_mask = mi.full((max_len, max_len), np.inf)
        self.register_buffer('attn_mask', attn_mask, persistable=False)

    def forward(self, x, seqs_len=1024):
        batch_size, src_len, d_model = x.shape

        return self.attn_mask[:, :, :seqs_len, :seqs_len]
        

class AxisNorm(nn.Layer):
    def __init__(self, axis=-1, epsilon=1e-6):
        super(AxisNorm, self).__init__()
        self.axis = axis
        self.epsilon = 1e-6

    def forward(self, x):
        x -= mi.mean(x, axis=self.axis, keepdim=True)
        x /= mi.sqrt(mi.var(x, axis=self.axis, keepdim=True) + self.epsilon)
        return x


class MyEmbeddingLayer(nn.Layer):
    def __init__(self, args, in_features=None):
        super(MyEmbeddingLayer, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        self.residue_fmt = args.residue_fmt

        if in_features is None:
            self.in_features = int(args.feature_dim)
        else:
            self.in_features = int(in_features)

        self.embed_dim = int(args.embed_dim)
        self.embed_num = int(args.embed_num)

        in_features = self.in_features # keep record of current feature dim
        if self.residue_fmt in ['scalar', 'quant'] and self.embed_num > 0:
            self.embed = nn.Embedding(
                in_features,
                self.embed_dim,
                padding_idx = 0,
                sparse = False)
            in_features = self.embed_dim
        else:
            pass # another option is

        self.out_features = in_features

    def forward(self, x, seqs_len=None):

        if hasattr(self, 'embed'):
            if not isinstance(x, mi.Tensor) or x.dtype.name != 'INT64':
                x = mi.to_tensor(x, dtype='int64')

            x = self.embed(x)
        else:
            if not isinstance(x, mi.Tensor) or x.dtype.name != 'FP32':
                x = mi.to_tensor(x, dtype='float32')

        return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = (4, 512, self.in_features)
        return mi.summary(self, input_size)


def MyLinearBlock(ndims, data_format='NLC', act_fn='ReLU', norm_fn='none', norm_axis=-1, dropout=0,
                    is_return=False):
    """ return a list of linear layers from ndims[0] to ndims[-1] """

    num_layers = len(ndims) # including the first and last layers
    data_format = data_format.upper()
    act_fn = act_fn.lower()
    norm_fn = norm_fn.lower()

    block_layers = []
    for idx_out in range(1, num_layers): # i_in = i_out -1
        block_layers.append(nn.Linear(ndims[idx_out - 1], ndims[idx_out]))

        if is_return and idx_out == num_layers - 1:
            break

        if act_fn == 'none':
            pass
        elif act_fn == 'relu':
            block_layers.append(nn.ReLU())
        elif act_fn == 'relu6':
            block_layers.append(nn.ReLU6())
        else:
            logger.warning(f'cannot recognize act_fn: {act_fn}')

        if norm_fn.startswith('none'):
            pass
        elif norm_fn.startswith('batch'):
            # for each dim along [C], norm_fn the [NL] 2D array
            block_layers.append(nn.BatchNorm1D(ndims[idx_out], data_format=data_format))
        elif norm_fn.startswith('insta'): # only works for NCL or NC format
            # for each dim along [C], normalize the [L] 1D array (no normalization along N)
            # InstanceNorm2D will normalize for [HW] for each channel
            block_layers.append(nn.InstanceNorm1D(ndims[idx_out], data_format=data_format))
        elif norm_fn.startswith('layer'):
            # normalize the ndarray specified by the passed shape, starting from the last dim
            # an integer will normalize the [:,...:,j] for each j for each data in the batch
            # a shape af a two-element tuple will normalize the last two dims
            # the passed shape must match the shapes of the data, starting from the last dim
            block_layers.append(nn.LayerNorm(ndims[idx_out]))
        elif norm_fn.startswith('axis'):
            block_layers.append(AxisNorm(norm_axis))
        else:
            logger.warning(f'cannot recognize norm_fn: {norm_fn}')

        if dropout > 0:
            block_layers.append(nn.Dropout(dropout, name=f'Dropout{dropout:0.2g}'))

    return block_layers


def MyConv1DBlock(ndims, stride=1, dilation=1, kernel_size=3, padding=1, padding_mode='zeros',
            max_pool=1, act_fn='relu', norm_fn='none', norm_axis=-1, dropout=0, data_format='NLC',
            is_return=False):
    """ return a list of conv1d layers from nchannels[0] to nchannels[-1] """

    num_layers = len(ndims) # including the first and last layers
    data_format = data_format.upper()
    act_fn = act_fn.lower()
    norm_fn = norm_fn.lower()

    block_layers = []
    for idx_out in range(1, num_layers): # i_in = i_out - 1
        block_layers.append(nn.Conv1D(
            in_channels = ndims[idx_out - 1],
            out_channels = ndims[idx_out],
            stride = stride,
            kernel_size = kernel_size,
            dilation = dilation,
            padding = padding,
            padding_mode = padding_mode,
            data_format = data_format,
        ))

        if is_return and idx_out == num_layers - 1:
            break

        if max_pool > 1:
            block_layers.append(nn.MaxPool1D(ndims[idx_out], stride=1, padding=max_pool // 2))

        if act_fn == 'none':
            pass
        elif act_fn == 'relu':
            block_layers.append(nn.ReLU())
        elif act_fn == 'relu6':
            block_layers.append(nn.ReLU6())
        else:
            logger.warning(f'cannot recognize act_fn: {act_fn}')

        if norm_fn.startswith('none'):
            pass
        elif norm_fn.startswith('batch'):
            block_layers.append(nn.BatchNorm1D(ndims[idx_out], data_format=data_format))
        elif norm_fn.startswith('insta'):
            block_layers.append(nn.InstanceNorm1D(ndims[idx_out], data_format=data_format))
        elif norm_fn.startswith('layer'):
            block_layers.append(nn.LayerNorm(ndims[idx_out]))
        elif norm_fn.startswith('axis'):
            block_layers.append(AxisNorm(norm_axis))
        else:
            logger.warning(f'cannot recognize norm_fn: {norm_fn}')

        if dropout > 0:
            block_layers.append(nn.Dropout(dropout))

    return block_layers


def MyConv2DBlock(ndims, stride=1, dilation=1, kernel_size=3, padding=1, padding_mode='zeros',
            max_pool=1, act_fn='relu', norm_fn='none', norm_axis=-1, dropout=0, data_format='NCHW',
            is_return=False):
    """ return a list of conv2d layers from nchannels[0] to nchannels[-1] """

    num_layers = len(ndims) # including the first and last layers
    data_format = data_format.upper()
    act_fn = act_fn.lower()
    norm_fn = norm_fn.lower()

    block_layers = []
    for idx_out in range(1, num_layers):
        block_layers.append(nn.Conv2D(
            in_channels = ndims[idx_out - 1],
            out_channels = ndims[idx_out],
            stride = stride,
            kernel_size = kernel_size,
            dilation = dilation,
            padding = padding,
            padding_mode = padding_mode,
            data_format = data_format,
        ))

        if is_return and idx_out == num_layers - 1:
            break

        if max_pool > 1:
            block_layers.append(nn.MaxPool2D(ndims[idx_out],
                    stride=1, padding=max_pool // 2,
                    data_format=data_format))

        if act_fn == 'none':
            pass
        elif act_fn == 'relu':
            block_layers.append(nn.ReLU())
        elif act_fn == 'relu6':
            block_layers.append(nn.ReLU6())
        else:
            logger.warning(f'cannot recognize act_fn: {act_fn}')

        if norm_fn.startswith('none'):
            pass
        elif norm_fn.startswith('batch'):
            block_layers.append(nn.BatchNorm2D(ndims[idx_out], data_format=data_format))
        elif norm_fn.startswith('insta'):
            block_layers.append(nn.InstanceNorm2D(ndims[idx_out], data_format=data_format))
        elif norm_fn.startswith('layer'):
            block_layers.append(nn.LayerNorm(ndims[idx_out]))
        elif norm_fn.startswith('axis'):
            block_layers.append(AxisNorm(norm_axis))
        else:
            logger.warning(f'cannot recognize norm_fn: {norm_fn}')

        if dropout > 0:
            block_layers.append(nn.Dropout2D(dropout, data_format=data_format))

    return block_layers


class MyLinearTower(nn.Layer):
    def __init__(self, args, in_features=None, is_return=False):
        """ is_return:True will trun off Act/Norm/Dropout for the last block """
        super(MyLinearTower, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        self.is_return = is_return
        if in_features is None:
            self.in_features = int(args.feature_dim)
        else:
            self.in_features = int(in_features)

        self.data_format = 'NLC'
        self.act_fn = args.act_fn
        self.norm_fn = args.norm_fn
        self.norm_axis = int(args.norm_axis)
        self.dropout = float(args.dropout)

        self.linear_dim = [int(_i) for _i in args.linear_dim] \
                if hasattr(args.linear_dim, '__len__') else [int(args.linear_dim)]
        self.linear_num = int(args.linear_num)
        self.linear_resnet = args.linear_resnet

        in_features = self.in_features
        self.linear_layers = [] # addditional layers if needed
        for i in range(self.linear_num):
            is_return = (i == self.linear_num -1) if self.is_return else False
            self.linear_layers.append(nn.Sequential(*MyLinearBlock(
                    [in_features] + self.linear_dim,
                    dropout = self.dropout,
                    act_fn = self.act_fn,
                    norm_fn = self.norm_fn,
                    norm_axis = self.norm_axis,
                    data_format = self.data_format,
                    is_return = is_return,
            )))

            if self.linear_resnet and in_features != self.linear_dim[-1]:
                logger.critical(f'linear_resnet requires in_features: {in_features} == linear_dim[-1]: {self.linear_dim[-1]}')

            in_features = self.linear_dim[-1]
            self.add_sublayer(f'leg1_linear{i}', self.linear_layers[i])

        self.out_features = in_features

    def forward(self, x, seqs_len=None):
        # if not isinstance(x, mi.Tensor) or x.dtype.name != 'FP32':
        #     x = mi.to_tensor(x, dtype='float32')

        for linear in self.linear_layers:
            if self.linear_resnet:
                x = x + linear(x)
            else:
                x = linear(x)

        return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = (4, 512, self.in_features)
        return mi.summary(self, input_size)


class MyLSTMTower(nn.Layer):
    def __init__(self, args, in_features=None):
        super(MyLSTMTower, self).__init__()

        self.dropout = float(args.dropout)

        if in_features is None:
            self.in_features = int(args.feature_dim)
        else:
            self.in_features = int(in_features)

        self.lstm_dim = [int(_i) for _i in args.lstm_dim] \
                if hasattr(args.lstm_dim, '__len__') else [int(args.lstm_dim)]
        self.lstm_direct = int(args.lstm_direct)
        self.lstm_num = int(args.lstm_num)
        self.lstm_resnet = args.lstm_resnet

        in_features = self.in_features
        self.lstm_layers = []
        for i in range(len(self.lstm_dim)):
            self.lstm_layers.append(nn.LSTM(
                input_size = in_features,
                hidden_size = self.lstm_dim[i],
                num_layers = self.lstm_num,
                direction = 'forward' if self.lstm_direct == 1 else 'bidirectional',
                dropout = self.dropout,
            ))

            out_features = self.lstm_dim[i] * self.lstm_direct
            if self.lstm_resnet and in_features != out_features:
                logger.critical(f'lstm_resnet requires in_features: {in_features} == out_features {out_features}')

            in_features = out_features
            self.add_sublayer(f'lstm{i}', self.lstm_layers[i])

        self.out_features = in_features

    def forward(self, x, seqs_len=None):
        # if not isinstance(x, mi.Tensor) or x.dtype.name != 'FP32':
        #     x = mi.to_tensor(x, dtype='float32')

        for lstm in self.lstm_layers:
            if self.lstm_resnet:
                x_out, (_, _) = lstm(x, initial_states=None, sequence_length=seqs_len)
                x = x + x_out
            else:
                x, (_, _) = lstm(x, initial_states=None, sequence_length=seqs_len)

        return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = (4, 512, self.in_features)
        return mi.summary(self, input_size)


class MyAttnTower(nn.Layer):
    def __init__(self, args, in_features=None):
        super(MyAttnTower, self).__init__()

        self.dropout = float(args.dropout)
        self.act_fn = args.attn_act

        if in_features is None:
            self.in_features = int(args.feature_dim)
        else:
            self.in_features = int(in_features)

        self.attn_num = int(args.attn_num)
        # self.attn_dim = int(args.attn_dim) # which is the same as in_features
        self.attn_ffdim = int(args.attn_ffdim)
        self.attn_nhead = int(args.attn_nhead)
        self.attn_dropout = args.attn_dropout # can be None
        self.attn_ffdropout = args.attn_ffdropout

        in_features = self.in_features

        self.posi_encoder = PositionEncoder((1, 2000, in_features))
        # self.attn_mask = AttentionMask(args.batch_size, self.attn_nhead, args.max_seqlen)

        attn_layer = nn.TransformerEncoderLayer(
            d_model = in_features,
            nhead = self.attn_nhead,
            dim_feedforward = self.attn_ffdim, # feed_forward dimension
            dropout = self.dropout, # between layers (default: 0.1)
            activation = self.act_fn, # (default: relu)
            attn_dropout = self.attn_dropout, # for self-attention target
            act_dropout = self.attn_ffdropout, # after activation in feedforward
            normalize_before = True, # between layers (appears important for upp prediction)
            weight_attr = None,
            bias_attr = None,
        )

        self.attn = nn.TransformerEncoder(
            attn_layer,
            num_layers= self.attn_num,
            norm = None,
        )

        self.out_features = in_features

    def forward(self, x, seqs_len=None):
        x = self.posi_encoder(x, beta=1.0)
        x = self.attn(x, get_attn_mask(x, seqs_len))
        return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = (4, 512, self.in_features)
        return mi.summary(self, input_size)


class MyConv1DTower(nn.Layer):
    def __init__(self, args, in_features=None):
        super(MyConv1DTower, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        self.data_format = 'NLC'

        self.act_fn = args.act_fn
        self.norm_fn = args.norm_fn
        self.norm_axis = int(args.norm_axis)
        self.dropout = float(args.dropout)


        if in_features is None:
            self.in_features = int(args.feature_dim)
        else:
            self.in_features = int(in_features)

        self.conv1d_dim = [int(_i) for _i in args.conv1d_dim]  \
                if hasattr(args.conv1d_dim, '__len__') else [int(args.conv1d_dim)]
        self.conv1d_num = int(args.conv1d_num)
        self.conv1d_resnet = args.conv1d_resnet

        self.conv1d_stride = 1
        self.conv1d_dilation = 1
        self.kernel_size = 5
        # self.padding =
        in_features = self.in_features

        # 1D convolution layers
        stride, dilation, kernel_size = 1, 1, 5
        # padding is set to return length/stride
        padding = calc_padding(kernel_size, stride=stride, dilation=dilation)

        self.conv1d_layers = []
        for i in range(self.conv1d_num):
            self.conv1d_layers.append(nn.Sequential(*MyConv1DBlock(
                    [in_features] + self.conv1d_dim,
                    stride = stride,
                    kernel_size = kernel_size,
                    dilation = dilation,
                    padding = padding, padding_mode = 'zeros',
                    data_format = self.data_format,
                    dropout = self.dropout,
                    act_fn = self.act_fn,
                    norm_fn = self.norm_fn,
                    norm_axis = self.norm_axis,
            )))

            if self.conv1d_resnet and in_features != self.conv1d_dim[-1]:
                logger.critical(f'conv1d_resnet requires in_features: {in_features} == conv1d_dim[-1]: {self.conv1d_dim[-1]}')

            in_features = self.conv1d_dim[-1]
            self.add_sublayer(f'conv1d{i}', self.conv1d_layers[i])

        self.out_features = in_features

    def forward(self, x, seqs_len=None):
        # if not isinstance(x, mi.Tensor) or x.dtype.name != 'FP32':
        #     x = mi.to_tensor(x, dtype='float32')

        for conv1d in self.conv1d_layers:
            if self.conv1d_resnet:
                x = x + conv1d(x)
            else:
                x = conv1d(x)

        return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = (4, 512, self.in_features)
        return mi.summary(self, input_size)


class MyConv2DTower(nn.Layer):
    def __init__(self, args, in_features=None):
        super(MyConv2DTower, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        self.act_fn = args.act_fn
        self.norm_fn = args.norm_fn
        self.norm_axis = int(args.norm_axis)
        self.dropout = float(args.dropout)
        self.data_format = 'NLC'

        if in_features is None:
            self.in_features = int(args.feature_dim)
        else:
            self.in_features = int(in_features)

        self.conv2d_dim = [int(_i) for _i in args.conv2d_dim]  \
                if hasattr(args.conv2d_dim, '__len__') else [int(args.conv2d_dim)]
        self.conv2d_num = int(args.conv2d_num)
        self.conv2d_resnet = args.conv2d_resnet

        in_features = self.in_features

        # 1D convolution layers
        stride, dilation, kernel_size = 1, 1, 3
        # padding is set to return length/stride
        padding = calc_padding(kernel_size, stride=stride, dilation=dilation)

        self.conv2d_layers = []
        for i in range(self.conv2d_num):
            self.conv2d_layers.append(nn.Sequential(*MyConv2DBlock(
                [in_features] + self.conv2d_dim,
                stride = stride,
                kernel_size = kernel_size,
                dilation = dilation,
                padding = padding,
                padding_mode = 'zeros',
                act_fn = self.act_fn,
                norm_fn = self.norm_fn,
                norm_axis = self.norm_axis,
                dropout = self.dropout,
                data_format = 'NCHW'
            )))
            if self.conv2d_resnet and in_features != self.conv2d_dim[-1]:
                logger.critical(f'conv2d_resnet requires in_features: {in_features} == conv2d_dim[-1]: {self.conv2d_dim[-1]}')

            in_features = self.conv2d_dim[-1]
            self.add_sublayer(f'conv2d{i}', self.conv2d_layers[i])

        self.out_features = in_features

    def forward(self, x, seqs_len=None):
        # if not isinstance(x, mi.Tensor) or x.dtype.name != 'FP32':
        #     x = mi.to_tensor(x, dtype='float32')

        for conv2d in self.conv2d_layers:
            if self.conv2d_resnet:
                x = x + conv2d(x)
            else:
                x = conv2d(x)
        return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = (4, 512, self.in_features)
        return mi.summary(self, input_size)


class Seq2MatTransform(nn.Layer):
    def __init__(self, method='concat', in_fmt='NCL', out_fmt='NCHW'):
        super(Seq2MatTransform, self).__init__()

        self.method = method.upper()
        self.in_fmt = in_fmt.upper()
        self.out_fmt = out_fmt.upper()

    def forward(self, xh, xw):
        if self.in_fmt == 'NCL':
            pass
        elif self.in_fmt == 'NLC':
            xh = xh.transpose([0, 2, 1])
            xw = xw.transpose([0, 2, 1])
        else:
            logger.critical(f"Uknown in_fmt: {self.in_fmt}!")

        N, C, H = xh.shape
        N1, C1, W = xw.shape
        assert N == N1, f"Two matrices must have the same N: {N} != {N1}!"

        xh = xh.unsqueeze(3).expand((N, C, H, W))
        xw = xw.unsqueeze(2).expand((N, C1, H, W))

        if self.method.startswith('CONCAT'):
            x = mi.concat([xh, xw], axis=1) # --> [N, C+C1, L, L]
        elif self.method.startswith('ADD'):
            assert C == C1, f"Cannot add two matrices with different C: {C} != {C1}"
            x = xh + xw
        elif self.method.startswith('MUL'):
            assert C == C1, f"Cannot multiply two matrices with different C: {C} != {C1}"
            x = xh * xw
        else:
            logger.critical(f"Unknown method: {self.method}")

        if self.out_fmt == 'NCHW':
            pass
        elif self.out_fmt == 'NHWC':
            x = x.transpose([0, 2, 3, 1])
        else:
            logger.critical(f"Uknown out_fmt: {self.out_fmt}!")

        return x


class LazyLinearNet(nn.Layer):
    """ This ignores all inter-residue interactions  """
    def __init__(self, args):
        super(LazyLinearNet, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        self.data_format = args.input_fmt.upper()
        self.feature_dim = int(args.feature_dim)

        in_features = self.feature_dim # keep record of current feature dim

        self.embed = MyEmbeddingLayer(args, in_features=in_features)
        in_features = self.embed.out_features

        self.linear_in = MyLinearTower(args, in_features=in_features)
        in_features = self.linear_in.out_features

        self.output_dim = [int(_i) for _i in args.output_dim]  \
                if hasattr(args.output_dim, '__len__') else [int(args.output_dim)]
        self.output_num = int(args.output_num)

        self.out = MyLinearTower(misc.Struct(
                data_format = args.input_fmt,
                feature_dim = args.feature_dim, # overwritten by in_features below
                linear_dim = args.output_dim,
                linear_num = args.output_num,
                linear_resnet = False,
                act_fn = args.act_fn,
                norm_fn = args.norm_fn,
                norm_axis = args.norm_axis,
                dropout = args.dropout,
                ),
                in_features = in_features,
                is_return = True,
        )

        # self.act_fn = args.act_fn
        # self.norm_fn = args.norm_fn
        # self.norm_axis = int(args.norm_axis)
        # self.dropout = float(args.dropout)

        # self.data_format = 'NLC'
        # self.feature_dim = args.feature_dim

        # self.linear_dim = [int(_i) for _i in args.linear_dim] \
        #         if hasattr(args.linear_dim, '__len__') else [int(args.linear_dim)]
        # self.linear_num = int(args.linear_num)
        # self.linear_resnet = args.linear_resnet

        # # the first layer to use feature_dim as the 1st dimension
        # in_features = self.feature_dim
        # self.leg1_linear = [] # addditional layers if needed
        # for i in range(self.linear_num):
        #     self.leg1_linear.append(nn.Sequential(*MyLinearBlock(
        #             [in_features] + self.linear_dim,
        #             dropout = self.dropout,
        #             act_fn = self.act_fn,
        #             norm_fn = self.norm_fn,
        #             norm_axis = self.norm_axis,
        #             data_format = self.data_format
        #     )))
        #     in_features = self.linear_dim[-1]

        #     self.add_sublayer(f'leg1_linear{i}', self.leg1_linear[i])
        #     # setattr(self, f'blk1layer{i}', self.blk1_linear[i])

        # self.out = nn.Sequential(
        #     nn.Linear(in_features=in_features, out_features=2),
        #     # nn.ReLU(),
        #     nn.Softmax(axis=-1),
        # )

    def summary(self, input_size=None):
        if input_size is None:
            if hasattr(self.embed, 'embed'):
                input_size = (4, 512)
            else:
                input_size = (4, 512, self.feature_dim)

        return mi.summary(self, input_size)

    # @mi.jit.to_static
    def forward(self, x, seqs_len=None):

        x = self.embed(x)
        x = self.linear_in(x, seqs_len=seqs_len)
        x = self.out(x)

        # if predict:
            # x = mi.squeeze(x[:,:,1], axis=-1)
            # if self.norm_axis is not None:
            #     logger.debug(f'applying axis norm for axis: {self.norm_axis}')
            #     x -= mi.mean(x, axis=self.norm_axis, keepdim=True)
            #     x /= mi.sqrt(mi.var(x, axis=self.norm_axis, keepdim=True) + 1e-6)

        return x


class Seq2Seq_LSTMNet(nn.Layer):
    def __init__(self, args):
        super(Seq2Seq_LSTMNet, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        self.data_format = args.input_fmt.upper()
        self.feature_dim = int(args.feature_dim)

        in_features = self.feature_dim # keep record of current feature dim

        self.embed = MyEmbeddingLayer(args, in_features=in_features)
        in_features = self.embed.out_features

        self.linear_in = MyLinearTower(args, in_features=in_features)
        in_features = self.linear_in.out_features

        self.lstm = MyLSTMTower(args, in_features=in_features)
        in_features = self.lstm.out_features

        self.output_dim = [int(_i) for _i in args.output_dim]  \
                if hasattr(args.output_dim, '__len__') else [int(args.output_dim)]
        self.output_num = int(args.output_num)

        self.out = MyLinearTower(misc.Struct(
                data_format = args.input_fmt,
                feature_dim = args.feature_dim, # overwritten by in_features below
                linear_dim = args.output_dim,
                linear_num = args.output_num,
                linear_resnet = False,
                act_fn = args.act_fn,
                norm_fn = args.norm_fn,
                norm_axis = args.norm_axis,
                dropout = args.dropout,
                ),
                in_features = in_features,
                is_return = True,
        )

    # @property
    def summary(self, input_size=None):
        if input_size is None:
            if hasattr(self.embed, 'embed'):
                input_size = (4, 512)
            else:
                input_size = (4, 512, self.feature_dim)

        return mi.summary(self, input_size)

    def forward(self, x, seqs_len=None):

        logger.debug('Applying self.embed()')
        x = self.embed(x)
        logger.debug('Applying self.linear_in()')
        x = self.linear_in(x, seqs_len=seqs_len)
        logger.debug('Applying self.lstm()')
        x = self.lstm(x, seqs_len=seqs_len)
        logger.debug('Applying self.out()')
        x = self.out(x)
        return x
        # return mi.squeeze(x, axis=-1)


class Seq2Seq_Conv1DNet(nn.Layer):
    """ This information  """
    def __init__(self, args):
        super(Seq2Seq_Conv1DNet, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        self.data_format = args.input_fmt.upper()
        self.feature_dim = int(args.feature_dim)

        in_features = self.feature_dim # keep record of current feature dim

        self.embed = MyEmbeddingLayer(args, in_features=in_features)
        in_features = self.embed.out_features

        self.linear_in = MyLinearTower(args, in_features=in_features)
        in_features = self.linear_in.out_features

        self.conv1d = MyConv1DTower(args, in_features=in_features)
        in_features = self.conv1d.out_features

        self.output_dim = [int(_i) for _i in args.output_dim]  \
                if hasattr(args.output_dim, '__len__') else [int(args.output_dim)]
        self.output_num = int(args.output_num)

        self.out = MyLinearTower(misc.Struct(
                data_format = args.input_fmt,
                feature_dim = args.feature_dim, # overwritten by in_features below
                linear_dim = args.output_dim,
                linear_num = args.output_num,
                linear_resnet = False,
                act_fn = args.act_fn,
                norm_fn = args.norm_fn,
                norm_axis = args.norm_axis,
                dropout = args.dropout,
                ),
                in_features = in_features,
                is_return = True,
        )

    def forward(self, x, seqs_len=None): #, predict=False):

        x = self.embed(x)
        x = self.linear_in(x, seqs_len=seqs_len)
        x = self.conv1d(x, seqs_len=seqs_len)
        x = self.out(x)

        # if predict:
            # x = mi.squeeze(x[:,:,1], axis=-1)

        return x

    def summary(self, input_size=None):
        if input_size is None:
            if hasattr(self.embed, 'embed'):
                input_size = (4, 512)
            else:
                input_size = (4, 512, self.feature_dim)

        return mi.summary(self, input_size)


class Seq2Seq_Conv2DNet(nn.Layer):
    """ This ignores all inter-residue information  """
    def __init__(self, args):
        super(Seq2Seq_Conv2DNet, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        self.act_fn = args.act_fn
        self.norm_fn = args.norm_fn
        self.norm_axis = args.norm_axis
        self.dropout = float(args.dropout)
        self.norm_axis = int(args.norm_axis)

        self.data_format = 'NLC'
        self.feature_dim = int(args.feature_dim)

        self.linear_dim = [int(_i) for _i in args.linear_dim]  \
                if hasattr(args.linear_dim, '__len__') else [int(args.linear_dim)]
        self.linear_num = int(args.linear_num)
        self.linear_resnet = args.linear_resnet

        self.conv2d_dim = [int(_i) for _i in args.conv2d_dim]  \
                if hasattr(args.conv2d_dim, '__len__') else [int(args.conv2d_dim)]
        self.conv2d_num = int(args.conv2d_num)
        self.conv2d_resnum = args.conv2d_resnet

        in_features = self.feature_dim # keep record of current feature dim

        in_features = self.feature_dim # keep record of current feature dim
        self.leg1_linear = [] # addditional layers if needed
        for i in range(self.linear_num):
            self.leg1_linear.append(nn.Sequential(*MyLinearBlock(
                    [in_features] + self.linear_dim,
                    dropout = self.dropout,
                    norm_fn = self.norm_fn,
                    act_fn = self.act_fn,
                    data_format = self.data_format
            )))
            in_features = self.linear_dim[-1]

            self.add_sublayer(f'leg1_linear{i}', self.leg1_linear[i])

        stride, dilation, kernel_size = 1, 1, 5
        # padding is calculated so as to return length/stride
        padding = calc_padding(kernel_size, stride=stride, dilation=dilation)

        in_features = in_features * 2 # due to outer concatenation

        self.leg2_conv2d = []
        for i in range(self.conv2d_num):
            self.leg2_conv2d.append(nn.Sequential(*MyConv2DBlock(
                [in_features] + self.conv2d_dim,
                stride = stride, kernel_size = kernel_size, dilation = dilation,
                padding = padding, padding_mode = 'zeros', norm_fn=self.norm_fn,
                dropout = self.dropout, data_format = 'NCHW'
            )))
            in_features = self.conv2d_dim[-1]

            self.add_sublayer(f'leg2_conv2d{i}', self.leg2_conv2d[i])

        self.leg3_linear = []
        for i in range(2):
            self.leg3_linear.append(nn.Sequential(*MyLinearBlock(
                [in_features, in_features], #, feature_dim // 2],
                dropout = self.dropout,
                act_fn = self.act_fn,
                norm_fn = self.norm_fn,
                data_format = self.data_format,
            )))
            in_features = in_features

            self.add_sublayer(f'leg3_linear{i}', self.leg3_linear[i])
            # setattr(self, f'blk3layer{i}', self.blk3_linear[i])

        self.out = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=2),
            # nn.ReLU(),
            nn.Softmax(axis=-1),
        )

    def forward(self, x, seqs_len=None):
        if not isinstance(x, mi.Tensor) or x.dtype.name != 'FP32':
            x = mi.to_tensor(x, dtype='float32')

        # x starts with [N:batch_size, L:seq_len, C:channel/feature_dim]
        # [N, L, C] --> [N, L, self.linear_dim[-1]]
        for linear in self.leg1_linear:
            if self.linear_resnet:
                x = x + linear(x)
            else:
                x = linear(x)

        # for each channel/feature, get a LxL matrix
        x = mi.transpose(x, perm=[0, 2, 1]) # [NLC] --> [NCL]
        new_shape = [x.shape[0], x.shape[1], x.shape[2], x.shape[2]]
        x = mi.concat([mi.broadcast_to(mi.unsqueeze(x, axis=3), shape=new_shape),
                       mi.broadcast_to(mi.unsqueeze(x, axis=2), shape=new_shape)],
                       axis=1) # [NCLL] --> [N, 2*C, L, L]

        for conv2d in self.leg2_conv2d:
            if self.conv2d_resnet:
                x = x + conv2d(x)
            else:
                x = conv2d(x)

        x = mi.transpose(x, perm=[0, 3, 2, 1]) # --> [N, L, L, 2*conv2d_dim[-1]]
        for linear in self.leg3_linear:
            x = linear(x)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2

        x = self.out(x)

        x = mi.squeeze(x[:, :, :, 0], axis=-1) # -> [N, L, L, 1] -> [NLL]

        x = x * (1.0 - mi.eye(x.shape[1], dtype='float32'))

        x = mi.max(x, axis=-1)

        # how to go from the LxL matrix to the unpaired probability
        # x = F.sigmoid(mi.sum(x, axis=-1))
        # concatenate or multiply (which reduces the feature dimension to 1)
        # x = mi.bmm(x, mi.transpose(x, perm=[0, 2, 1]))
        return x # mi.squeeze(x[:,:,0], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = (4, 512, self.feature_dim)
        return mi.summary(self, input_size)


class Seq2Seq_AttnNet(nn.Layer):
    def __init__(self, args):
        super(Seq2Seq_AttnNet, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        self.data_format = args.input_fmt.upper()
        self.feature_dim = int(args.feature_dim)

        in_features = self.feature_dim # keep record of current feature dim

        self.embed = MyEmbeddingLayer(args, in_features=in_features)
        in_features = self.embed.out_features

        self.linear_in = MyLinearTower(args, in_features=in_features)
        in_features = self.linear_in.out_features

        self.attn = MyAttnTower(args, in_features=in_features)
        in_features = self.attn.out_features

        self.output_dim = [int(_i) for _i in args.output_dim]  \
                if hasattr(args.output_dim, '__len__') else [int(args.output_dim)]
        self.output_num = int(args.output_num)

        self.out = MyLinearTower(misc.Struct(
                data_format = args.input_fmt,
                feature_dim = args.feature_dim, # overwritten by in_features below
                linear_dim = args.output_dim,
                linear_num = args.output_num,
                linear_resnet = False,
                act_fn = args.act_fn,
                norm_fn = args.norm_fn,
                norm_axis = args.norm_axis,
                dropout = args.dropout,
                ),
                in_features = in_features,
                is_return = True,
        )

    def forward(self, x, seqs_len=None):

        x = self.embed(x)
        x = self.linear_in(x, seqs_len=seqs_len)
        x = self.attn(x, seqs_len=seqs_len)
        x = self.out(x)

        return x

    # @property
    def summary(self, input_size=None):
        if input_size is None:
            if hasattr(self.embed, 'embed'):
                input_size = (2, 512)
            else:
                input_size = (2, 512, self.feature_dim)
        return mi.summary(self, input_size)


class Seq2Seq_EmbedLSTMNet_OLD(nn.Layer):
    def __init__(self, args):
        super(Seq2Seq_EmbedLSTMNet_OLD, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        self.act_fn = args.act_fn
        self.norm_fn = args.norm_fn
        self.norm_axis = int(args.norm_axis)
        self.dropout = float(args.dropout)

        self.data_format = 'NLC'
        self.feature_dim = int(args.feature_dim)
        self.embed_dim = int(args.embed_dim)

        self.linear_dim = [int(_i) for _i in args.linear_dim]  \
                if hasattr(args.linear_dim, '__len__') else [int(args.linear_dim)]
        self.linear_num = int(args.linear_num)
        self.linear_resnet = args.linear_resnet

        self.lstm_dim = [int(_i) for _i in args.lstm_dim] \
                if hasattr(args.lstm_dim, '__len__') else [int(args.lstm_dim)]
        self.lstm_direct = int(args.lstm_direct)
        self.lstm_num = int(args.lstm_num)
        self.lstm_resnet = args.lstm_resnet

        in_features = self.feature_dim # keep record of current feature dim
        self.embed = nn.Embedding(
                    in_features,
                    self.embed_dim,
                    padding_idx = 0,
                    sparse = True)
        in_features = self.embed_dim

        self.leg1_linear = [] # addditional layers if needed
        for i in range(self.linear_num):
            self.leg1_linear.append(nn.Sequential(*MyLinearBlock(
                    [in_features] + self.linear_dim,
                    dropout = self.dropout,
                    norm_fn = self.norm_fn,
                    norm_axis = self.norm_axis,
                    act_fn = self.act_fn,
                    data_format = self.data_format
            )))
            in_features = self.linear_dim[-1]

            self.add_sublayer(f'leg1_linear{i}', self.leg1_linear[i])
            # setattr(self, f'blk1layer{i}', self.blk1_linear[i])

        # how to give the initial hidden and cell states of lstm???
        # Maybe it is not important
        self.leg2_lstm = []
        for i in range(len(self.lstm_dim)):
            self.leg2_lstm.append(nn.LSTM(
                input_size = in_features,
                hidden_size = self.lstm_dim[i],
                num_layers = self.lstm_num,
                direction = 'forward' if self.lstm_direct == 1 else 'bidirectional',
                dropout = args.dropout,
            ))
            in_features = self.lstm_dim[i] * self.lstm_direct

            self.add_sublayer(f'leg2_lstm{i}', self.leg2_lstm[i])
            # setattr(self, f'blk2layer{i}', self.blk2_lstm[i])

        self.leg3_linear = []
        for i in range(2):
            self.leg3_linear.append(nn.Sequential(*MyLinearBlock(
                [in_features, in_features // 2], #, feature_dim // 2],
                dropout = self.dropout,
                act_fn = self.act_fn,
                norm_fn = self.norm_fn,
                norm_axis = self.norm_axis,
                data_format = self.data_format,
            )))
            in_features = in_features // 2

            self.add_sublayer(f'leg3_linear{i}', self.leg3_linear[i])
            # setattr(self, f'blk3layer{i}', self.blk3_linear[i])

        self.out = nn.Sequential(
            nn.Linear(in_features, 2),
            # nn.ReLU(),
            nn.Softmax(axis=-1),
        )

    # @property
    def summary(self, input_size=None):
        if input_size is None:
            input_size = (2, 512)
        return mi.summary(self, input_size)

    def forward(self, x, seqs_len=None):
        if not isinstance(x, mi.Tensor) or x.dtype.name != 'INT64':
            x = mi.to_tensor(x, dtype='int64')

        x = self.embed(x)

        for linear in self.leg1_linear:
            if self.linear_resnet:
                x = x + linear(x)
            else:
                x = linear(x)

        # x = mi.concat((x, F.relu(self.conv1(x))), axis=-1)
        for lstm in self.leg2_lstm:
            if self.lstm_resnet:
                x_out, (_, _) = lstm(x, initial_states=None, sequence_length=seqs_len)
                x = x + x_out
            else:
                x, (_, _) = lstm(x, initial_states=None, sequence_length=seqs_len)

        for linear in self.leg3_linear:
            x = linear(x)

        x = self.out(x)
        # return mi.squeeze(x, axis=-1)
        return mi.squeeze(x[:,:,0], axis=-1)


class Seq2Seq_EmbedAttnNet_OLD(nn.Layer):
    def __init__(self, args):
        super(Seq2Seq_EmbedAttnNet_OLD, self).__init__()

        self.act_fn = args.act_fn
        self.norm_fn = args.norm_fn
        self.norm_axis = int(args.norm_axis)
        self.dropout = float(args.dropout)

        self.data_format = 'NLC'
        self.feature_dim = args.feature_dim
        self.embed_dim = int(args.embed_dim)

        self.linear_dim = [int(_i) for _i in args.linear_dim]  \
                if hasattr(args.linear_dim, '__len__') else [int(args.linear_dim)]
        self.linear_num = int(args.linear_num)
        self.linear_resnet = args.linear_resnet

        self.attn_num = int(args.attn_num)
        self.attn_nhead = int(args.attn_nhead)
        self.attn_act = args.attn_act
        # self.attn_dim = int(args.attn_dim)
        self.attn_dropout = args.attn_dropout # can be None
        self.attn_ffdim = int(args.attn_ffdim)
        self.attn_ffdropout = args.attn_ffdropout

        in_features = args.feature_dim

        self.embed = nn.Embedding(
            in_features,
            self.embed_dim,
            padding_idx = 0,
            sparse = True)
        in_features = self.embed_dim

        self.leg1_linear = [] # addditional layers if needed
        for i in range(self.linear_num):
            self.leg1_linear.append(nn.Sequential(*MyLinearBlock(
                    [in_features] + self.linear_dim,
                    dropout = self.dropout,
                    norm_fn = self.norm_fn,
                    norm_axis = self.norm_axis,
                    act_fn = self.act_fn,
                    data_format = self.data_format
            )))
            in_features = self.linear_dim[-1]

            self.add_sublayer(f'leg1_linear{i}', self.leg1_linear[i])
            # setattr(self, f'blk1layer{i}', self.blk1_linear[i])


        attn_layer = nn.TransformerEncoderLayer(
            d_model = in_features,
            nhead = self.attn_nhead,
            dim_feedforward = self.attn_ffdim, # feed_forward dimension
            dropout = self.dropout, # between layers (default: 0.1)
            activation = self.attn_act, # (default: relu)
            attn_dropout = self.attn_dropout, # for self-attention target
            act_dropout = self.attn_ffdropout, # after activation in feedforward
            normalize_before = False, # between layers
            weight_attr = None,
            bias_attr = None,
        )

        self.leg2_attn = nn.TransformerEncoder(attn_layer,
            num_layers=args.attn_num) # norm = args.norm_fn,

        self.leg3_linear = []
        for i in range(2):
            self.leg3_linear.append(nn.Sequential(*MyLinearBlock(
                [in_features, in_features], #, feature_dim // 2],
                dropout = self.dropout,
                act_fn = self.act_fn,
                norm_fn = self.norm_fn,
                data_format = self.data_format,
            )))
            in_features = in_features

            self.add_sublayer(f'leg3_linear{i}', self.leg3_linear[i])

        self.out = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=2),
            nn.Softmax(axis=-1),
        )

    def forward(self, x, seqs_len=None):
        if not isinstance(x, mi.Tensor) or x.dtype.name != 'INT64':
            x = mi.to_tensor(x, dtype='int64')

        x = self.embed(x)

        for linear in self.leg1_linear:
            if self.linear_resnet:
                x = x + linear(x)
            else:
                x = linear(x)

        x += position_encoding_trig(x.shape)
        x = self.leg2_attn(x)

        # x, (_, _) = self.lstm(x)
        for linear in self.leg3_linear:
            x = linear(x)

        x = self.out(x)
        return mi.squeeze(x[:,:,0], axis=-1)

    # @property
    def summary(self, input_size=None):
        input_size = (2, 512) if input_size is None else tuple(input_size)
        return mi.summary(self, input_size)


class Seq2Seq_Conv1DLSTMNet(nn.Layer):
    """ This information  """
    def __init__(self, args):
        super(Seq2Seq_Conv1DLSTMNet, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        self.data_format = args.input_fmt.upper()
        self.feature_dim = int(args.feature_dim)

        in_features = self.feature_dim # keep record of current feature dim

        self.embed = MyEmbeddingLayer(args, in_features=in_features)
        in_features = self.embed.out_features

        self.linear_in = MyLinearTower(args, in_features=in_features)
        in_features = self.linear_in.out_features

        self.conv1d = MyConv1DTower(args, in_features=in_features)
        in_features = self.conv1d.out_features

        self.lstm = MyLSTMTower(args, in_features=in_features)
        in_features = self.lstm.out_features

        self.output_dim = [int(_i) for _i in args.output_dim]  \
                if hasattr(args.output_dim, '__len__') else [int(args.output_dim)]
        self.output_num = int(args.output_num)

        self.out = MyLinearTower(misc.Struct(
                data_format = args.input_fmt,
                feature_dim = args.feature_dim, # overwritten by in_features below
                linear_dim = args.output_dim,
                linear_num = args.output_num,
                linear_resnet = False,
                act_fn = args.act_fn,
                norm_fn = args.norm_fn,
                norm_axis = args.norm_axis,
                dropout = args.dropout,
                ),
                in_features = in_features,
                is_return = True,
        )

    def forward(self, x, seqs_len=None):

        x = self.embed(x)
        x = self.linear_in(x, seqs_len=seqs_len)
        x = self.conv1d(x, seqs_len=seqs_len)
        x = self.lstm(x, seqs_len=seqs_len)
        x = self.out(x)

        return x

    def summary(self, input_size=None):
        if input_size is None:
            if hasattr(self.embed, 'embed'):
                input_size = (4, 512)
            else:
                input_size = (4, 512, self.feature_dim)
        return mi.summary(self, input_size)


class Seq2Seq_AttnLSTMNet(nn.Layer):
    def __init__(self, args):
        super(Seq2Seq_AttnLSTMNet, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        self.data_format = args.input_fmt.upper()
        self.feature_dim = int(args.feature_dim)

        in_features = self.feature_dim # keep record of current feature dim

        self.embed = MyEmbeddingLayer(args, in_features=in_features)
        in_features = self.embed.out_features

        self.linear_in = MyLinearTower(args, in_features=in_features)
        in_features = self.linear_in.out_features

        self.attn = MyAttnTower(args, in_features=in_features)
        in_features = self.attn.out_features

        self.lstm = MyLSTMTower(args, in_features=in_features)
        in_features = self.lstm.out_features

        self.output_dim = [int(_i) for _i in args.output_dim]  \
                if hasattr(args.output_dim, '__len__') else [int(args.output_dim)]
        self.output_num = int(args.output_num)

        self.out = MyLinearTower(misc.Struct(
                data_format = args.input_fmt,
                feature_dim = args.feature_dim, # overwritten by in_features below
                linear_dim = args.output_dim,
                linear_num = args.output_num,
                linear_resnet = False,
                act_fn = args.act_fn,
                norm_fn = args.norm_fn,
                norm_axis = args.norm_axis,
                dropout = args.dropout,
                ),
                in_features = in_features,
                is_return = True,
        )

    # @property
    def summary(self, input_size=None):
        if input_size is None:
            if hasattr(self.embed, 'embed'):
                input_size = (2, 512)
            else:
                input_size = (2, 512, self.feature_dim)
        return mi.summary(self, input_size)

    def forward(self, x, seqs_len=None):

        x = self.embed(x)
        x = self.linear_in(x, seqs_len=seqs_len)
        x = self.attn(x, seqs_len=seqs_len)
        x = self.lstm(x, seqs_len=seqs_len)

        x = self.out(x)

        return x


class Seq2Seq_AttnLSTMConv1DNet(nn.Layer):
    def __init__(self, args):
        super(Seq2Seq_AttnLSTMConv1DNet, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        self.data_format = args.input_fmt.upper()
        self.feature_dim = int(args.feature_dim)

        in_features = self.feature_dim # keep record of current feature dim

        self.embed = MyEmbeddingLayer(args, in_features=in_features)
        in_features = self.embed.out_features

        self.linear_in = MyLinearTower(args, in_features=in_features)
        in_features = self.linear_in.out_features

        self.attn = MyAttnTower(args, in_features=in_features)
        in_features = self.attn.out_features

        self.lstm = MyLSTMTower(args, in_features=in_features)
        in_features = self.lstm.out_features

        self.conv1d = MyConv1DTower(args, in_features=in_features)
        in_features = self.conv1d.out_features

        self.output_dim = [int(_i) for _i in args.output_dim]  \
                if hasattr(args.output_dim, '__len__') else [int(args.output_dim)]
        self.output_num = int(args.output_num)

        self.out = MyLinearTower(misc.Struct(
                data_format = args.input_fmt,
                feature_dim = args.feature_dim, # overwritten by in_features below
                linear_dim = args.output_dim,
                linear_num = args.output_num,
                linear_resnet = False,
                act_fn = args.act_fn,
                norm_fn = args.norm_fn,
                norm_axis = args.norm_axis,
                dropout = args.dropout,
                ),
                in_features = in_features,
                is_return = True,
        )

    # @property
    def summary(self, input_size=None):
        if input_size is None:
            if hasattr(self.embed, 'embed'):
                input_size = (2, 512)
            else:
                input_size = (2, 512, self.feature_dim)
        return mi.summary(self, input_size)

    def forward(self, x, seqs_len=None):

        x = self.embed(x)
        x = self.linear_in(x, seqs_len=seqs_len)
        x = self.attn(x, seqs_len=seqs_len)
        x = self.lstm(x, seqs_len=seqs_len)
        x = self.conv1d(x)

        x = self.out(x)

        return x


class Seq2Mat_Conv2DNet(nn.Layer):
    """ This ignores all inter-residue information  """
    def __init__(self, args):
        super(Seq2Mat_Conv2DNet, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        self.act_fn = args.act_fn
        self.norm_fn = args.norm_fn
        self.dropout = float(args.dropout)
        self.norm_axis = int(args.norm_axis)

        self.data_format = 'NLC'
        self.feature_dim = int(args.feature_dim)

        in_features = self.feature_dim # keep record of current feature dim

        self.embed = MyEmbeddingLayer(args, in_features=in_features)
        in_features = self.embed.out_features

        self.linear_in = MyLinearTower(args, in_features=in_features)
        in_features = self.linear_in.out_features

        in_features = in_features * 2 # due to outer concatenation
        self.conv2d = MyConv2DTower(args, in_features=in_features)
        in_features = self.conv2d.out_features

        self.linear_out_list = []
        for i in range(2):
            self.linear_out_list.append(nn.Sequential(*MyLinearBlock(
                [in_features, in_features], #, feature_dim // 2],
                dropout = self.dropout,
                act_fn = self.act_fn,
                norm_fn = self.norm_fn,
                data_format = self.data_format,
            )))
            in_features = in_features

            self.add_sublayer(f'leg3_linear{i}', self.linear_out_list[i])
            # setattr(self, f'blk3layer{i}', self.blk3_linear[i])

        self.out = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=2),
            # nn.ReLU(),
            # nn.Softmax(axis=-1),
        )

    def forward(self, x, seqs_len=None):
        # x starts with [N:batch_size, L:seq_len, C:channel/feature_dim]
        # [N, L, C] --> [N, L, self.linear_dim[-1]]

        x = self.embed(x)
        x = self.linear_in(x, seqs_len=seqs_len)

        # for each channel/feature, get a LxL matrix
        x = mi.transpose(x, perm=[0, 2, 1]) # [NLC] --> [NCL]
        new_shape = [x.shape[0], x.shape[1], x.shape[2], x.shape[2]]
        x = mi.concat([mi.broadcast_to(mi.unsqueeze(x, axis=3), shape=new_shape),
                       mi.broadcast_to(mi.unsqueeze(x, axis=2), shape=new_shape)],
                       axis=1) # [NCLL] --> [N, 2*C, L, L]

        x = self.conv2d(x, seqs_len=seqs_len)

        x = mi.transpose(x, perm=[0, 3, 2, 1]) # --> [N, L, L, 2*conv2d_dim[-1]]

        for linear in self.linear_out_list:
            x = linear(x)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2

        x = self.out(x)

        # x = mi.squeeze(x[:, :, :, 0], axis=-1) # -> [N, L, L, 1] -> [NLL]

        # x = x * (1.0 - mi.eye(x.shape[1], dtype='float32'))

        # x = mi.max(x, axis=-1)

        # how to go from the LxL matrix to the unpaired probability
        # x = F.sigmoid(mi.sum(x, axis=-1))
        # concatenate or multiply (which reduces the feature dimension to 1)
        # x = mi.bmm(x, mi.transpose(x, perm=[0, 2, 1]))
        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            if hasattr(self.embed, 'embed'):
                input_size = (4, 512)
            else:
                input_size = (4, 512, self.feature_dim)
        return mi.summary(self, input_size)


class Seq2Mat_LSTMConv2DNet(nn.Layer):
    """ This ignores all inter-residue information  """
    def __init__(self, args):
        super(Seq2Mat_LSTMConv2DNet, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        self.act_fn = args.act_fn
        self.norm_fn = args.norm_fn
        self.dropout = float(args.dropout)
        self.norm_axis = int(args.norm_axis)

        self.data_format = 'NLC'
        self.feature_dim = int(args.feature_dim)

        in_features = self.feature_dim # keep record of current feature dim

        self.embed = MyEmbeddingLayer(args, in_features=in_features)
        in_features = self.embed.out_features

        self.linear_in = MyLinearTower(args, in_features=in_features)
        in_features = self.linear_in.out_features

        self.lstm = MyLSTMTower(args, in_features=in_features)
        in_features = self.lstm.out_features

        in_features = in_features * 2 # due to outer concatenation
        self.conv2d = MyConv2DTower(args, in_features=in_features)
        in_features = self.conv2d.out_features

        self.linear_out_list = []
        for i in range(2):
            self.linear_out_list.append(nn.Sequential(*MyLinearBlock(
                [in_features, in_features], #, feature_dim // 2],
                dropout = self.dropout,
                act_fn = self.act_fn,
                norm_fn = self.norm_fn,
                data_format = self.data_format,
            )))
            in_features = in_features

            self.add_sublayer(f'linear_out{i}', self.linear_out_list[i])
            # setattr(self, f'blk3layer{i}', self.blk3_linear[i])

        self.out = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=2),
            # nn.ReLU(),
            # nn.Softmax(axis=-1),
        )

    def forward(self, x, seqs_len=None):

        x = self.embed(x)
        x = self.linear_in(x, seqs_len=seqs_len)
        x = self.lstm(x, seqs_len=seqs_len)

        # for each channel/feature, get a LxL matrix
        x = mi.transpose(x, perm=[0, 2, 1]) # [NLC] --> [NCL]
        new_shape = [x.shape[0], x.shape[1], x.shape[2], x.shape[2]]
        x = mi.concat([mi.broadcast_to(mi.unsqueeze(x, axis=3), shape=new_shape),
                       mi.broadcast_to(mi.unsqueeze(x, axis=2), shape=new_shape)],
                       axis=1) # [NCLL] --> [N, 2*C, L, L]

        x = self.conv2d(x, seqs_len=seqs_len)

        x = mi.transpose(x, perm=[0, 3, 2, 1]) # --> [N, L, L, 2*conv2d_dim[-1]]
        for linear in self.linear_out_list:
            x = linear(x)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2

        x = self.out(x)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            if hasattr(self.embed, 'embed'):
                input_size = (4, 512)
            else:
                input_size = (4, 512, self.feature_dim)
        return mi.summary(self, input_size)


class Seq2Mat_Conv1DLSTMConv2DNet(nn.Layer):
    """ This ignores all inter-residue information  """
    def __init__(self, args):
        super(Seq2Mat_Conv1DLSTMConv2DNet, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        self.act_fn = args.act_fn
        self.norm_fn = args.norm_fn
        self.dropout = float(args.dropout)
        self.norm_axis = int(args.norm_axis)

        self.data_format = 'NLC'
        self.feature_dim = int(args.feature_dim)

        in_features = self.feature_dim # keep record of current feature dim

        self.embed = MyEmbeddingLayer(args, in_features=in_features)
        in_features = self.embed.out_features

        self.linear_in = MyLinearTower(args, in_features=in_features)
        in_features = self.linear_in.out_features

        self.conv1d = MyConv1DTower(args, in_features=in_features)
        in_features = self.conv1d.out_features

        self.lstm = MyLSTMTower(args, in_features=in_features)
        in_features = self.lstm.out_features

        in_features = in_features * 2 # due to outer concatenation
        self.conv2d = MyConv2DTower(args, in_features=in_features)
        in_features = self.conv2d.out_features

        self.linear_out_list = []
        for i in range(2):
            self.linear_out_list.append(nn.Sequential(*MyLinearBlock(
                [in_features, in_features], #, feature_dim // 2],
                dropout = self.dropout,
                act_fn = self.act_fn,
                norm_fn = self.norm_fn,
                data_format = self.data_format,
            )))
            in_features = in_features

            self.add_sublayer(f'linear_out{i}', self.linear_out_list[i])
            # setattr(self, f'blk3layer{i}', self.blk3_linear[i])

        self.out = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=2),
            # nn.ReLU(),
            # nn.Softmax(axis=-1),
        )

    def forward(self, x, seqs_len=None):

        x = self.embed(x)
        x = self.linear_in(x, seqs_len=seqs_len)
        x = self.conv1d(x, seqs_len=seqs_len)
        x = self.lstm(x, seqs_len=seqs_len)

        # for each channel/feature, get a LxL matrix
        x = mi.transpose(x, perm=[0, 2, 1]) # [NLC] --> [NCL]
        new_shape = [x.shape[0], x.shape[1], x.shape[2], x.shape[2]]
        x = mi.concat([mi.broadcast_to(mi.unsqueeze(x, axis=3), shape=new_shape),
                       mi.broadcast_to(mi.unsqueeze(x, axis=2), shape=new_shape)],
                       axis=1) # [NCLL] --> [N, 2*C, L, L]

        x = self.conv2d(x, seqs_len=seqs_len)

        x = mi.transpose(x, perm=[0, 3, 2, 1]) # --> [N, L, L, 2*conv2d_dim[-1]]
        for linear in self.linear_out_list:
            x = linear(x)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2

        x = self.out(x)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            if hasattr(self.embed, 'embed'):
                input_size = (4, 512)
            else:
                input_size = (4, 512, self.feature_dim)
        return mi.summary(self, input_size)
