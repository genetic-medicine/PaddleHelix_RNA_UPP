#!/usr/bin/env python
# external
import os
import sys
import argparse
import logging
import time
import pickle # the same as _pickle
import itertools
import importlib
import shutil
import numpy as np
import pandas as pd
import multiprocessing as mpi
from datetime import datetime
from inspect import getmembers, isclass
from pathlib import Path as path
from functools import partial as func_partial
from tqdm import tqdm

import paddle as mi
import paddle.nn as nn
from paddle.io import Dataset
import paddle.nn.functional as F
from sklearn.model_selection import train_test_split

# homebrew
import misc
import gwio
import mol_stru
import paddle_nets as MiNets

logger = logging.getLogger(__name__)

def parse_args2(*argv):
    """ argv is in the format of sys.argv[1:]
    Also serves the purpose to get default args 
    """
    parser = argparse.ArgumentParser(description='''
        Launch Pad for PaddlePaddle''',
        formatter_class=argparse.RawTextHelpFormatter)

    paparser = argparse.ArgumentParser(add_help=False)
    # paparser.add_argument('--action', type=str, nargs='+', default=[], metavar='', help="the action to take")
    paparser.add_argument("--argv", type=str, default='-h', help='commandline argv (auto defined)')
    paparser.add_argument('-v', '--verbose', choices=[0,1,2], default=1, type=int, metavar='', help="('_')")
    paparser.add_argument("--resume", action='store_true', help='whether to load model state dicts')
    paparser.add_argument('--load_dir', type=str, default=None, metavar='', help="directory for loading model args and states")
    paparser.add_argument('--save_dir', type=str, default=None, metavar='', help="directory for saving model and results")
    paparser.add_argument('--save_level', type=int, default=2, metavar='', help="0: no save, 1: final only, 2: interim")
    paparser.add_argument('--save_grpby', type=str, nargs='+', default=['epoch', 'batch'], metavar='', help="groupby columns, e.g., batch, epoch")
    paparser.add_argument('--log', type=str, default=path(__file__).stem + f'-{datetime.now().strftime("%b%d")}.log', metavar='', help="the log file")

    # data
    paparser.add_argument('--data_args', type=str, default='======= data args =======', metavar='', help="======= data args =======")
    paparser.add_argument('--data_dir', type=str, default='data', metavar='', help="data directory")
    paparser.add_argument('--data_name', type=str, default=None, metavar='', help="data file name")
    paparser.add_argument('--data_suffix', type=str, default='.pkl', metavar='', help="data file suffix")
    paparser.add_argument('--data_size', type=int, default=0, metavar='', help="the number of data to use if > 0")
    paparser.add_argument('--test_size', type=float, default=0.1, metavar='', help="for train_test_split()")
    paparser.add_argument('--split_seed', type=int, default=None, metavar='', help="not used yet")

    paparser.add_argument('--input_genre', type=str, default='Seq', metavar='', help="not used yet")
    paparser.add_argument('--input_fmt', type=str, default='NLC', metavar='', help="")
    paparser.add_argument('--seq_length', type=int, nargs='+', default=[0, 512, -1], metavar='', help="<0: use max data len, 0: no padding")
    paparser.add_argument('--residue_fmt', type=str, default='vector', metavar='', help="embed/vector or quant/scalar")
    paparser.add_argument('--residue_nn', type=int, default=0, metavar='', help="# of nearest neighbors to use")
    paparser.add_argument('--residue_dbn', action='store_true', help="use residue dbn in data")
    paparser.add_argument('--residue_attr', action='store_true', help="use residue attribute in data")
    paparser.add_argument('--residue_extra', action='store_true', help="use residue extra features in data")

    paparser.add_argument('--label_genre', type=str, default='upp', metavar='', help="data type: upp/ct/dist/...")
    paparser.add_argument('--label_fmt', type=str, default='NL', help="not yet used")
    paparser.add_argument('--label_tone', type=str, default='none', help="none/hard/soft")
    paparser.add_argument('--label_ntype', type=int, default=2, help="number of label types")
    paparser.add_argument('--label_smooth', action='store_true', help="whether to smoothen labels")

    # net
    paparser.add_argument('--net_args', type=str, default='======= net args =======', metavar='', help="======= net args =======")
    paparser.add_argument('--net_src_file', type=str, default=path(MiNets.__file__), metavar='', help="")
    paparser.add_argument('--net', type=str, default='lazylinear', metavar='', help="the name of the net class")
    paparser.add_argument('--resnet', action='store_true', help="whether to use residual net")
    # paparser.add_argument('--net_return', type=str, default='upp', metavar='', help="")
    paparser.add_argument('--act_fn', type=str, default='relu', metavar='', help="activation: relu/sigmoid/...")
    paparser.add_argument('--norm_fn', type=str, default='none', metavar='', help="normalization: batch/instance/layer/none")
    paparser.add_argument('--norm_axis', type=int, default=-1, metavar='', help="the axis # when norm_fn is axis")
    paparser.add_argument('--dropout', type=float, default=0.2, metavar='', help="the dropout fraction")

    paparser.add_argument('--feature_dim', type=int, default=1, metavar='', help="the size of in_features")
    paparser.add_argument('--embed_dim', type=int, default=32, metavar='', help="specific for embedding")
    paparser.add_argument('--embed_num', type=int, default=1, metavar='', help="0: no, 1: yes, if residue_fmt is scalar/quant")

    paparser.add_argument('--linear_num', type=int, default=2, metavar='', help="# of linear layers")
    paparser.add_argument('--linear_dim', type=int, nargs='+', default=[32], metavar='', help="dims of linear layers")
    paparser.add_argument('--linear_resnet', action='store_true', help="whether to use residual net")

    paparser.add_argument('--conv1d_num', type=int, default=1, metavar='', help="# of Conv1D layers")
    paparser.add_argument('--conv1d_dim', type=int, nargs='+', default=[32], metavar='', help="channels of Conv1D layers")
    paparser.add_argument('--conv1d_resnet', action='store_true', help="whether to use residual net")
    paparser.add_argument('--conv1d_stride', type=int, default=1, metavar='', help="stride in 1D convolution")

    paparser.add_argument('--conv2d_num', type=int, default=1, metavar='', help="# of Conv2D layers")
    paparser.add_argument('--conv2d_dim', type=int, nargs='+', default=[32], metavar='', help="channels of Conv2D layers ")
    paparser.add_argument('--conv2d_resnet', action='store_true', help="whether to use residual net")

    paparser.add_argument('--attn_num', type=int, default=2, metavar='', help="# of Attention Encoder layers")
    paparser.add_argument('--attn_nhead', type=int, default=2, metavar='', help="# of heads of Attention Encoders")
    paparser.add_argument('--attn_act', type=str, default='relu', metavar='', help="activation of Attention Encoders")
    paparser.add_argument('--attn_dropout', type=float, default=None, metavar='', help="dropout of Attention Encoders")
    paparser.add_argument('--attn_ffdim', type=int, default=32, metavar='', help="feedforward dims of Attention Encoders")
    paparser.add_argument('--attn_ffdropout', type=float, default=None, metavar='', help="feedforward dropout of Attention Encoders")

    paparser.add_argument('--lstm_num', type=int, default=2, metavar='', help="# of LSTM layers")
    paparser.add_argument('--lstm_dim', type=int, nargs='+', default=[32], metavar='', help="dims of LSTM layers")
    paparser.add_argument('--lstm_direct', type=int, default=2, metavar='', help="direction of LSTM layers")
    paparser.add_argument('--lstm_resnet', action='store_true', help="whether to use residual net")

    # paparser.add_argument('--output_net', type=str, default='linear', metavar='', help="not used yet")
    paparser.add_argument('--output_num', type=int, default=1, metavar='', help="# of output layers")
    paparser.add_argument('--output_dim', type=int, nargs='+', default=[32,32,2], metavar='', help="output hidden dimensions")
    paparser.add_argument('--output_resnet', action='store_true', help="whether to use residual net (should never set it!)")

    ## optimization
    paparser.add_argument('--optim_args', type=str, default='======= optim args =======', metavar='', help="======= optim args =======")
    paparser.add_argument('--optim', type=str, default='adam', metavar='', help="optimizer type: adam/sgd/...")
    paparser.add_argument('--learning_rate', type=float, default=0.003, metavar='', help="change to 3e-4???")
    paparser.add_argument('--beta1', type=float, default=0.9, metavar='', help="beta1 for Adam")
    paparser.add_argument('--beta2', type=float, default=0.999, metavar='', help="beta2 for Adam")
    paparser.add_argument('--epsilon', type=float, default=1e-8, metavar='', help="epsilon for Adam")

    paparser.add_argument('--lr_scheduler', type=str, default='reduced', metavar='', help="learning rate scheduler")
    paparser.add_argument('--lr_factor', type=float, default=0.9, metavar='', help="learning rate relative change factor")
    paparser.add_argument('--lr_patience', type=int, default=10, metavar='', help="learning rate patience")

    paparser.add_argument('--weight_decay', type=str, default='none', metavar='', help="weight decay: l1 or l2")
    paparser.add_argument('--l1decay', type=float, default=1e-4, metavar='', help="L1Decay rate")
    paparser.add_argument('--l2decay', type=float, default=1e-4, metavar='', help="L2Decay rate")

    paparser.add_argument('--train_args', type=str, default='======= train/loss args =======', metavar='', help="======= train/loss args =======")
    paparser.add_argument('--batch_size', type=int, default=4, metavar='', help="for train/validate/predict")
    # paparser.add_argument('--dynamic_loader', action='store_true', help="turn on dynamic loading")
    paparser.add_argument('--num_epochs', type=int, default=777, metavar='', help="# of maximum epochs (may be terminated early)")
    paparser.add_argument('--num_recaps_per_epoch', type=int, default=30, metavar='', help="# of recaps/summaries per epoch")
    paparser.add_argument('--num_callbacks_per_epoch', type=int, default=10, metavar='', help="# of validation callbacks per epoch")

    paparser.add_argument('--loss_fn', type=str, nargs='+', default=['mse'], metavar='', help="loss function type: mse/bce/...")
    paparser.add_argument('--loss_fn_scale', type=float, nargs='+', default=[1.0], metavar='', help="")
    paparser.add_argument('--loss_sqrt', action='store_true', help="take sqrt before summing losses in a batch")
    paparser.add_argument('--loss_padding', action='store_true', help="include padded seqs/zeros in the loss")

    paparser.add_argument('--validate_callback', type=str, default=None, metavar='', help="not used yet")
    paparser.add_argument('--trainloss_rdiff', type=float, default=1e-3, metavar='', help="relative difference")
    paparser.add_argument('--validloss_rdiff', type=float, default=1e-3, metavar='', help="relative difference")
    paparser.add_argument('--trainloss_patience', type=int, default=11, metavar='', help="train loss change patience")
    paparser.add_argument('--validloss_patience', type=int, default=11, metavar='', help="valid loss change patience")

    # pre-defined settings
    paparser.add_argument('--mood_args', type=str, default='======= mood args =======', metavar='', help="======= mood args =======")
    paparser.add_argument('--debug', action='store_true', help="minimal hidden units")
    paparser.add_argument('--lucky', action='store_true', help="[256, 256]")
    paparser.add_argument('--lazy', action='store_true', help="[64]*3")
    paparser.add_argument('--sharp', action='store_true', help="[64]*3")
    paparser.add_argument('--comfort', action='store_true', help="[128]*3]")
    paparser.add_argument('--explore', action='store_true', help="[512]")
    paparser.add_argument('--exploit', action='store_true', help="[32]*5]")
    paparser.add_argument('--diehard', action='store_true', help="[512]*7")
    paparser.add_argument('--tune', action='store_true', help="('_')")

    paparser.add_argument('--action_args', type=str, default='======= action args =======', metavar='', help="======= action args =======")

    # action as subparsers
    subparsers = parser.add_subparsers(dest='action', required=True)

    subparser = subparsers.add_parser('summary', parents=[paparser], description='', help='only view net/loss')
    subparser = subparsers.add_parser('summarize', parents=[paparser], description='', help='alias for summary')
    subparser = subparsers.add_parser('view', parents=[paparser], description='', help='alias for summary')

    subparser = subparsers.add_parser('train', parents=[paparser], description='', help='just do it')
    subparser.set_defaults() # it will be overwritten by later set_defaults!

    subparser = subparsers.add_parser('dynamic_train', parents=[paparser], description='', help='not implemented')
    subparser.set_defaults() # it will be overwritten by later set_defaults!
    subparser.add_argument("--seq_lengths", nargs='+', type=int, default=[300, 800, 2000, 5000], metavar='', help='a list of integers')
    subparser.add_argument("--batch_sizes", nargs='+', type=int, default=[8, 4, 2, 1], metavar='', help='a list of integers')

    subparser = subparsers.add_parser('cross_validate', parents=[paparser], description='', help='cross validate the model')
    subparser.set_defaults()
    subparser.add_argument("--num_cvs", type=int, default=5, metavar='', help='# of cross validations')

    subparser = subparsers.add_parser('validate', parents=[paparser], description='', help='predict and calculate loss')
    subparser.set_defaults()

    subparser = subparsers.add_parser('predict', parents=[paparser], description='', help='predict and save')
    subparser.set_defaults(data_name='predict')

    subparser = subparsers.add_parser('average_model', parents=[paparser], description='average multiple models', help='')
    subparser.set_defaults()
    subparser.add_argument("--model_dirs", type=str, nargs='+', default=[], metavar='', help='directories of the models to average')
    subparser.add_argument("--best_earlystop", action='store_true', help='use the best earlystop states in each directory')

    subparser = subparsers.add_parser('scan_data', parents=[paparser], description='', help='scan data_size and batch_size')
    subparser.set_defaults()
    subparser.add_argument("--data_sizes", nargs='+', type=int, default=[0], metavar='', help='a list of integers')
    subparser.add_argument("--batch_sizes", nargs='+', type=int, default=[1,2,4,8], metavar='', help='a list of integers')

    subparser = subparsers.add_parser('scout_args', parents=[paparser], description='', help='scout model args')
    subparser.set_defaults()
    subparser.add_argument("--rebake_midata", action='store_true', help='whether new midata need to be obtained for each iter')
    subparser.add_argument("--grid_search", action='store_true', help='perform grid search')
    subparser.add_argument("--spawn_search", action='store_true', help='perform spawn search')

    subparser.add_argument("--num_scouts", type=int, default=7, metavar='', help='only need for grid_search=False')
    subparser.add_argument("--num_spawns", type=int, default=7, metavar='', help='only need for spawn_search=True')

    subparser.add_argument("--arg_names", nargs='+', type=str, default=['learning_rate', 'dropout'], metavar='', help='a list of strings')
    subparser.add_argument("--arg_values", nargs='+', type=str, default=['0.0001,0.001,0.01', '0.1,0.3,0.5'],
                metavar='', help='a list of STRINGs with values separated by "," for each arg\n' +
                'if grid_search is true, each string contains all values for the arg\n' +
                'if not grid_search, the string format is "min,max"')
    subparser.add_argument("--arg_scales", nargs='+', type=int, default=[0], metavar='',
                help='the scale for each arg_names, 0 for linear, log otherwise')

    # args are managed in loosely defined three tiers
    #    1) the default values in parse_args([]), which runs without any user args
    #    2) the loaded args from args.load_dir (if applicable)
    #    3) the user_args from command line, parsed by and extracted from parse_args()
    # the ruling order is the 3 overwrites 2 overwrites 1, which is why user_args needs to be returned!
    #
    # Note: ONLY ONE set of args is maintained for each model
    if isinstance(argv, str): argv = [argv]
    argv = misc.unpack_list_tuple(argv)
    args = misc.Struct(vars(parser.parse_args(argv)))
    argv_dict = vars(misc.argv_optargs(argv, args)) # remember command line args

    return args, argv_dict


def autoconfig_args(args):
    """ set default arg values not easily done with argparser """
    ####### DATA #######
    # directories should be path
    if args.save_dir: args.save_dir = path(args.save_dir)
    if args.data_dir: args.data_dir = path(args.data_dir)

    if args.data_name is None:
        if args.action in ['validate']:
            args.data_name = 'valid'
        elif args.action in ['predict', 'average_model']:
            args.data_name = 'predict'
        else:
            args.data_name = 'train'

    # net determines residue_fmt
    # if 'embed' in args.net.lower():
    #     args.residue_fmt = 'quant'
    # else:
    #     args.residue_fmt = 'vector'

    # residue_fmt determine feature_dim
    args.residue_fmt = args.residue_fmt.lower()
    if args.residue_fmt in ['vector', 'embed']:
        args.feature_dim = 4 * (1 + 2 * args.residue_nn)
        if args.residue_dbn: args.feature_dim += 4 * (1 + 2 * args.residue_nn)
        if args.residue_attr: args.feature_dim += 8
        if args.residue_extra: args.feature_dim += 2
    elif args.residue_fmt in ['quant', 'scalar']:
        args.feature_dim = 5 ** (1 + 2 * args.residue_nn) # the number of possible values
        if args.residue_dbn:
            args.feature_dim *= 4 ** (1 + 2 * args.residue_nn)

    # >1 batch_size requires the same sequence length
    if args.seq_length[-1] == 0 and args.batch_size > 1:
        args.seq_length[-1] = -1

    args.label_genre = args.label_genre.lower()
    if args.label_genre in ['upp']:
        pass
    elif args.label_genre in ['ct']:
        args.residue_dbn = False

    # set smaller numbers for debug
    if args.debug:
        args.embed_dim = min([8, args.embed_dim])
        args.linear_dim = [min([16, i]) for i in args.linear_dim]
        args.linear_num = min([2, args.linear_num])
        args.conv1d_dim = [min([8, i]) for i in args.conv1d_dim]
        args.conv1d_num = min([1, args.conv1d_num])
        args.conv2d_dim = [min([8, i]) for i in args.conv2d_dim]
        args.conv2d_num = min([1, args.conv2d_num])
        args.lstm_num = min([1, args.lstm_num])
        args.lstm_dim = [min([8, i]) for i in args.lstm_dim]
        args.attn_num = min([2, args.attn_num])
        args.attn_ffdim = min([16, args.attn_num])
        args.output_dim = [min([8, i]) for i in args.output_dim]
        args.output_num = 1
        args.batch_size = min([2, args.batch_size])
        args.num_epochs = min([3, args.num_epochs])

    if args.lucky:
        args.linear_dim = [256, 256]
        args.linear_num = 1

    if args.sharp:
        args.linear_dim = [32]
        args.linear_num = 5

    if args.explore:
        args.dropout = 0.5
        args.learning_rate = 1e-3
        args.l1decay = 0
        args.l2decay = 0

    if args.tune:
        args.learning_rate = 5e-5
        args.l1decay = 0
        args.l2decay = 1e-4
    if args.exploit:
        pass

    return args


def random_sample(midata, size=1, replace=False):
    """ midata can be a list/tuple/np.ndarray, replace=True will yield repeated elements """
    return [midata[i] for i in np.random.choice(len(midata), size, replace=replace)]


def random_split_dict(dict_in, size=0.1):
    """ split each key values like train_test_split  """
    num_data = len(dict_in[list(dict_in.keys())[0]])

    # size can be a number or a fraction
    if 0.0 < size < 1.0:
        size = int(num_data * size)
    else:
        size = int(size)

    # make sure size < num_data / 2
    if size > num_data / 2:
        size = num_data - size
        reverse_order = True
    else:
        reverse_order = False

    indices = np.sort(np.random.choice(num_data, size, replace=False))

    dict_out1, dict_out2 = dict(), dict()

    for key, val in dict_in.items():
        dict_out1[key] = []
        dict_out2[key] = []

        dict_out2[key].extend(val[:indices[0]])

        for i in range(size - 1):
            dict_out1[key].append(val[indices[i]])
            dict_out2[key].extend(val[indices[i] + 1:indices[i + 1]])

        dict_out1[key].append(val[indices[-1]])
        dict_out2[key].extend(val[indices[-1] + 1:])

    if reverse_order:
        return dict_out2, dict_out1
    else:
        return dict_out1, dict_out2


def fix_length1d(data, length, **kwargs):
    """ np.pad is used for padding, the same kwargs """
    data_len = len(data)

    if data_len >= length:
        return data[:length]
    else:
        return np.pad(data, (0, length - data_len), 'constant', **kwargs)


def fix_length2d(data, length, **kwargs):
    """ np.pad is used for padding, the same kwargs """
    data_len = data.shape

    if isinstance(length, int) or isinstance(length, np.integer):
        length = [length]

    len2pad = [0, 0]

    if data_len[0] >= length[0]: # check 1st dimension
        data = data[:length[0], :]
    else:
        len2pad[0] = length[0] - data_len[0]

    if data_len[1] >= length[-1]: # check 2nd dimension
        data = data[:, :length[-1]]
    else:
        len2pad[1] = length[-1] - data_len[1]

    if any(len2pad): # pad if needed
        return np.pad(data, ((0, len2pad[0]), (0, len2pad[1])), 'constant', **kwargs)
    else:
        return data

def cut_padding(data, seq_len):
    """ assume the first dimension is batch_size and all data in the batch has the same seq_en"""
    if data.ndim == 1:
        data = data[:seq_len]
    elif data.ndim == 2:
        data = data[:, :seq_len]
    elif data.ndim == 3:
        data = data[:, :seq_len, :seq_len]
    elif data.ndim == 4:
        data = data[:, :seq_len, :seq_len, :seq_len]
    elif data.ndim == 5:
        data = data[:, :seq_len, :seq_len, :seq_len, :seq_len]
                
    return data

def soft2hard_label(data, keep_dim=False, np=np):
    """ convert soft to hard labels """
    hard_data = data.argmax(axis=-1)
    if keep_dim:
        return np.expand_dims(hard_data, -1)
    else:
        return hard_data


def hard2soft_label(data, nlabel=2, discrete=False, np=np):
    """ true label starts from zero
    accept non-integers for hard labels, in which weights are assigned to
    two neighboring classes depending on the distance
    """
    
    soft_data = np.zeros(list(data.shape) + [nlabel], dtype='float32')
    
    for i in range(nlabel): # there must be a better way...
        soft_data[..., i] = np.clip(np.abs(data - i), 0.0, 1.0)

    soft_data = 1.0 - soft_data

    return soft_data


def load_pkldata(args=misc.Struct(), **kwargs):
    """ kwargs > args > my_args, return pkldata, a dict """
    def_args = misc.Struct(dict(
        data_dir = '',
        data_name = 'train',
        data_suffix = '.pkl',
        verbose = 1,
    ))
    def_args.update(vars(args))
    def_args.update(kwargs)
    args.update(vars(def_args))

    # data_dir must be path
    data_dir = args.data_dir if isinstance(args.data_dir, path) \
        else path(args.data_dir)

    # fname must be string
    fname = args.data_name.name if isinstance(args.data_name, path) \
        else args.data_name

    # read the pkl file
    pkldata_file = (data_dir / fname).with_suffix(args.data_suffix)
    logger.info(f'Loading data: {pkldata_file}')
    with pkldata_file.open('rb') as hfile:
        pkldata = pickle.load(hfile) # it is a dictionary of id, seq, dbn, ct... when available

    # logger.info(f'    # of data: {len(pkldata["seq"])}, min len: {pkldata["len"].min()}, max len: {pkldata["len"].max()}')

    return pkldata


def bake_midata(in_pkldata, args=misc.Struct(), **kwargs):
    """ kwargs > args > my_args
    return midata, a list of dataset for each sample """
    def_args = misc.Struct(dict(
        label_genre = 'upp',
        label_tone = 'none',
        label_ntype = 2,
        seq_length = [-1],
        residue_fmt = 'embed', # "embed" a residue as a vector
        residue_nn = 0, # do not include nearest neighbor
        residue_dbn = False,
        residue_attr = False,
        residue_extra = False,
        verbose = 1,
    ))
    def_args.update(vars(args))
    def_args.update(kwargs)
    args.update(vars(def_args))

    # collect basic info about data
    num_seqs = len(in_pkldata['seq'])
    in_pkldata['len'] = np.array(in_pkldata['len'])
    args.max_seqlen = in_pkldata['len'].max()

    logger.info(f'   # of data: {num_seqs},  max seqlen: {args.max_seqlen}, user seq_length: {args.seq_length}')
    logger.info(f' residue fmt: {args.residue_fmt}, nn: {args.residue_nn}, dbn: {args.residue_dbn},' + \
                f' attr: {args.residue_attr}, genre: {args.label_genre}')

    # deal with sequence length
    #    1) seq_length as [min, max, pad_flag] or [max, pad_flag] (min would be zero)
    #           only select sequences/samples with len between [min, max]
    #    2) seq_length as [pad_flag]
    #           all sequences used
    # pad_flag is always the last element: >0: cut/pad to the length, 0: keep original, <0: use max_seqlen
    if type(args.seq_length) not in (list, tuple):
        args.seq_length = [args.seq_length]
    if len(args.seq_length) == 2: args.seq_length = [0] + args.seq_length
    if len(args.seq_length) > 2: # [min, max, pad_flag]
        pkldata = dict()
        seqs_idx = np.array([_i for _i, _len in enumerate(in_pkldata['len']) 
                    if args.seq_length[0] <= _len <= args.seq_length[1]], dtype=np.int32)
        for _key, _val in in_pkldata.items():
            pkldata[_key] = [_val[_i] for _i in seqs_idx]
        pkldata['len'] = np.array(pkldata['len'], dtype=np.int32)
        num_seqs = len(seqs_idx)
        args.max_seqlen = pkldata['len'].max()
        logger.info(f'Selected {num_seqs} data sets with length range: {args.seq_length}')
    else:
        pkldata = in_pkldata
        seqs_idx = np.arange(num_seqs, dtype=np.int32)

    if args.seq_length[-1] < 0: 
        seq_length = args.max_seqlen
    elif args.seq_length[-1] == 0:
        seq_length = 0
    else:
        seq_length = args.seq_length[-1]

    # process to get midata
    logger.debug('Using multiprocessing pool...')
    mpool = mpi.Pool(processes=int(mpi.cpu_count() * 0.8))

    # get "x": sequence data
    if args.residue_fmt in ['vector', 'embed']:
        embed_func = func_partial(mol_stru.vector_rna_seq, 
                    use_nn=args.residue_nn,
                    use_attr=args.residue_attr,
                    use_dbn=args.residue_dbn,
                    length=seq_length)
    elif args.residue_fmt in ['scalar', 'quant']:
        embed_func = func_partial(mol_stru.quant_rna_seq, 
                    use_nn=args.residue_nn,
                    use_dbn=args.residue_dbn,
                    length=seq_length)
    else:
        logger.critical(f'Unknown residue format: {args.residue_fmt}')

    if args.residue_dbn and 'dbn' in pkldata:
        seqdata = mpool.starmap(embed_func, zip(pkldata['seq'], pkldata['dbn']))
    else:
        seqdata = mpool.map(embed_func, pkldata['seq'])

    # add residue_extra to the seqdata (aka input)
    if args.residue_extra and 'extra' in pkldata:
        seqdata = [np.concatenate((seqdata[_i], 
                fix_length2d(pkldata['extra'][_i], [seqdata[_i].shape[0], pkldata['extra'][_i].shape[1]])),
                axis=1) for _i in range(len(seqdata))]

    # get length, idx (START FROM 1!!!)
    if seq_length > 0:
        seqs_len = pkldata['len'].copy()
        seqs_len[seqs_len > seq_length] = seq_length
    else:
        seqs_len = pkldata['len']

    lendata = np.concatenate((seqs_len.reshape((-1, 1)),
                              seqs_idx.reshape((-1, 1)) + 1), axis=1)

    # get "y": upp/ct/...
    uppdata = None
    if 'upp' in pkldata:
        logger.info('Processing upp data...')
        if seq_length == 0:
            uppdata = pkldata['upp']
        else:
            pad_func = func_partial(fix_length1d, constant_values=(0,0))
            uppdata = mpool.starmap(pad_func, zip(pkldata['upp'], [seq_length] * num_seqs))

    ctdata = None
    if 'ct' in pkldata:
        logger.info('Processing ct data...')
        ctdata = mpool.starmap(mol_stru.ct2mat, zip(pkldata['ct'], pkldata['len']))

        if seq_length > 0:
            pad_func = func_partial(fix_length2d, constant_values=((0, 0), (0, 0)))
            ctdata = mpool.starmap(pad_func, zip(ctdata, [seq_length] * num_seqs))

        # expand the last dimension
        # ctdata = [np.expand_dims(_ct, -1) for _ct in ctdata]

    mpool.close()

    # return
    midata = None
    args.label_genre = args.label_genre.lower()
    if args.label_genre == 'upp':
        if uppdata is None:
            midata = list(zip(seqdata, lendata))
        else:
            midata = list(zip(seqdata, lendata, uppdata))
    elif args.label_genre == 'ct':
        if ctdata is None:
            midata = list(zip(seqdata, lendata))
        else:
            midata = list(zip(seqdata, lendata, ctdata))

    if args.verbose > 1:
        shapes = [data.shape for data in midata[0]]
        print(f'Number of datasets: {len(midata)}')
        print(f'Number of items in each set: {len(midata[0])}, with shapes: {shapes}')
        print('Checking for consistent dimensions...')
        for i, data in enumerate(midata):
            for j, item in enumerate(data):
                if shapes[j] != item.shape:
                    print(f'The shape of dataset #{i} item #{j}: {item.shape} differs from the first: {shapes[j]}')
        print('Done!')

    return midata


def get_midata(args=misc.Struct(), **kwargs):
    """ kwargs > args > my_args
    return midata, meta_data (a pandas dataframe) """
    pkldata = load_pkldata(args, **kwargs)
    return bake_midata(pkldata, args, **kwargs)


class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(midata, **kwargs):
    """  """
    loader_opts = dict(
        shuffle = False,
        batch_size = 1,
        timeout = 0,
        num_workers = int(mpi.cpu_count() * 0.5),
        use_buffer_reader = True, # False will corrupt cuda version somehow
    )
    loader_opts.update(kwargs)

    return mi.io.DataLoader(MyDataset(midata), **loader_opts)


def get_net(args, quiet=False):
    """  """

    if isinstance(args.load_dir, str): args.load_dir = path(args.load_dir)
    if isinstance(args.net_src_file, str): args.net_src_file = path(args.net_src_file)

    # get the local src code path
    if args.load_dir and args.net_src_file:
        local_src_code = args.load_dir / args.net_src_file.name
    else:
        local_src_code = path(args.net_src_file)

    # reload the net if a local copy exits
    if local_src_code.exists() and os.path.exists(MiNets.__file__) and not local_src_code.samefile(MiNets.__file__):
        args.net_src_file = local_src_code
        logger.info(f'Found local net definition: {local_src_code}')
        sys.path.insert(0, local_src_code.parent.absolute().as_posix())
        if local_src_code.name == path(MiNets.__file__).name:
            LocalNets = importlib.reload(MiNets)
        else:
            LocalNets = importlib.import_module(local_src_code.stem)
        sys.path.remove(local_src_code.parent.absolute().as_posix())
    else:
        LocalNets = globals()['MiNets']

    logger.info(f'Used net definition: {misc.str_color(LocalNets.__file__, bkg="cyan")}')
    # locate net classes by name
    net_classes = getmembers(LocalNets, isclass)
    net_names = [_s[0].lower() for _s in net_classes]
    idx_net = misc.get_list_index(net_names, args.net.lower())
    if not idx_net:
        idx_net = misc.get_list_index(net_names, args.net.lower() + 'net')
    if not idx_net:
        logger.error(f'No net definition with name: {args.net} found!')
        return None

    # use the first match
    net_init = net_classes[idx_net[0]][1]
    upp_net = net_init(args)
    if not quiet and hasattr(upp_net, 'summary'):
        args.params = upp_net.summary()
        logger.info(f'{args.params}')

    return upp_net


def save_net_pycode(net_src_file, save_dir):
    """ not yet to get model save to work, so save the code! """
    if isinstance(net_src_file, str): net_src_file = path(net_src_file)
    if isinstance(save_dir, str): save_dir = path(save_dir)

    # net_src_file = path(MyNets.__file__)
    net_des_file = save_dir / net_src_file.name
    if net_src_file.resolve().as_posix() == net_des_file.resolve().as_posix():
        logger.info(f'Net python code: {net_des_file} aleady exists...')
    else:
        net_des_file.write_text(net_src_file.read_text())
        logger.info(f'Saved net python code: {net_des_file}')
    return net_des_file


def get_optimizer(upp_net, args):
    """ two returns: the optimizer and [lr_scheduler or learning_rate] """
    weight_decay = None
    if args.weight_decay.lower().startswith('l1'):
        weight_decay = mi.regularizer.L1Decay(args.l1decay)
    elif args.weight_decay.lower().startswith('l2'):
        weight_decay = mi.regularizer.L2Decay(args.l2decay)

    if args.lr_scheduler.lower().startswith('non'):
        learning_rate = args.learning_rate
    else:
        learning_rate = mi.optimizer.lr.ReduceOnPlateau(
            learning_rate = args.learning_rate,
            factor = args.lr_factor,
            patience = args.lr_patience,
            verbose = True,
        )

    upp_opt = mi.optimizer.Adam(
        parameters = upp_net.parameters(),
        weight_decay = weight_decay,
        learning_rate = learning_rate,
        beta1 = args.beta1,
        beta2 = args.beta2,
        epsilon = args.epsilon,
        )

    logger.info(f'Optimizer method: {args.optim}')
    logger.info(f'   learning rate: {args.learning_rate}')
    logger.info(f'    lr_scheduler: {args.lr_scheduler}')
    logger.info(f'    weight decay: {args.weight_decay}')
    logger.info(f'         l1decay: {args.l1decay}')
    logger.info(f'         l2decay: {args.l2decay}')

    return upp_opt, learning_rate


def f_score(input, label, batch=False, beta=1.0, epsilon=1e-7, **kwargs):
    """ beta is used to weight recall relative to precision """
    assert input.ndim == label.ndim, 'Input and Label must have the same # of dims!'

    if batch:
        sum_axis = list(range(1, input.ndim))
    else:
        sum_axis = list(range(0, input.ndim))

    tp = (input * label).sum(sum_axis)
    fp = (input * (1.0 - label)).sum(sum_axis)
    fn = ((1.0 - input) * label).sum(sum_axis)
    # tn = ((1 - input) * (1 - label)).sum()

    beta2 = beta * beta
    fscore = (1.0 + beta2) * tp / ((1.0 + beta2) * tp + beta2 * fp + fn + epsilon)
    return fscore


def sigmoid_mse(input, label, reduction='none'):
    input = F.sigmoid(input)
    loss = F.mse_loss(input, label, reduction=reduction)
    return loss


def softmax_mse(input, label, label_col=1, reduction='none'):
    """ this only makes sense for input.shape[-1]=2 
    label_col is only used if input.ndim == label.ndim + 1
    """

    input = F.softmax(input, axis=-1)
    
    # only take one axis for loss calculation
    if input.ndim == label.ndim + 1:
        if input.ndim == 2: # yet to find a better way, tensor doesn't accept [...,label_col]
            input = input[:, label_col].squeeze(-1)
        elif input.ndim == 3:
            input = input[:, :, label_col].squeeze(-1)
        elif input.ndim == 4:
            input = input[:, :, :, label_col].squeeze(-1)
        elif input.ndim == 5:
            input = input[:, :, :, :, label_col].squeeze(-1)
        elif input.ndim == 6:
            input = input[:, :, :, :, :, label_col].squeeze(-1)
        else:
            logger.critical(f'Feeling dizzy with too many dimensions: {input.ndim}!')

    loss = F.mse_loss(input, label, reduction=reduction)
    
    return loss


def softmax_bce(input, label, label_col=1, reduction='none'):
    """ this only makes sense for input.shape[-1]=2 """
    assert input.ndim == label.ndim + 1, \
            f"input.ndim:{input.ndim}- label.ndim:{label.ndim} != 1!"
    assert input.shape[-1] == 2,  \
            f"input.shape[-1]:{input.shape[-1]} != 2!"

    y0, y1 = mi.unstack(input, axis=-1)

    if label_col == 1: # which index to use to compare with the label
        y_delta = y1 - y0
    else:
        y_delta = y0 - y1

    # this is a reduced formula, please derive to check
    loss = mi.log(1.0 + mi.exp(y_delta)) - label * y_delta
    
    if reduction == 'mean':
        loss = loss.mean()
    
    return loss


class SeqLossFn_Agg(nn.Layer):
    """ calculate the aggregated loss for input vs. label 
    meant for loss_fn which cannot give point to point loss contributions
    """
    def __init__(self, fn=f_score, name='fscore', **kwargs):
        super(SeqLossFn_Agg, self).__init__()

        self.name = name.lower()
        self.fn = fn
        self.kwargs = kwargs

    def as_label(self, input):
        if self.name in ['fscore', 'f-score']:
            input = F.softmax(input, axis=-1)
            _, input = mi.unstack(input, axis=-1)
        else:
            logger.critical(f'Cannot recognize loss_fn name: {self.name}!')

        return input
        
    def forward(self, input, label, seqs_len=None, loss_padding=False, loss_sqrt=False, **kwargs):
        """ return the loss with the shape as input """

        fn_kwargs = self.kwargs.copy()
        fn_kwargs.update(kwargs)

        if self.name == 'fscore':
            # the data is supposed to be before softmax (last dimension with shape of 2), and label may have the final dimension as 1
            # input = mi.unstack(input, axis=-1)
            # input = mi.squeeze(mi.greater_equal(input[:,:,:,1], input[:,:,:,0] * 2.71828), axis=-1)
            # input = F.softmax(input, axis=-1)
            # input = mi.squeeze(input[:,:,:,1], axis=-1)
            # label = mi.squeeze(label, axis=-1)
            # if not isinstance(label, mi.Tensor) or label.dtype.name != 'INT64':
                # label = mi.to_tensor(label, dtype='int64')
            input = self.as_label(input)
        else:
            logger.critical(f'loss fn: {self.name} not supported yet!')

        batch_size = input.shape[0]

        logger.debug(f'     num_dims: {input.ndim}')
        logger.debug(f'    data size: {batch_size}')
        logger.debug(f'     seqs_len: {seqs_len is not None}')
        logger.debug(f' loss_padding: {loss_padding}')
        logger.debug(f'  loss kwargs: {fn_kwargs}')

        loss_for_backprop = mi.to_tensor(0.0, dtype='float32', stop_gradient=False)
        loss_vs_seq = np.zeros((batch_size), dtype=np.float32)
        std_vs_seq = np.zeros((batch_size), dtype=np.float32)
        for i in range(batch_size):
            seq_len = int(seqs_len[i])

            if loss_padding or input.ndim == 1: # process loss_mat as a whole
                seq_loss = self.fn(input[i], label[i], **fn_kwargs)
            elif input.ndim == 2:
                seq_loss = self.fn(input[i, :seq_len], label[i, :seq_len], **fn_kwargs)
            elif input.ndim == 3:
                seq_loss = self.fn(input[i, :seq_len, :seq_len],
                                   label[i, :seq_len, :seq_len],
                                   **fn_kwargs)
            elif input.ndim == 4:
                seq_loss = self.fn(input[i, :seq_len, :seq_len, :seq_len],
                                   label[i, :seq_len, :seq_len, :seq_len],
                                   **fn_kwargs)
            elif input.ndim == 5:
                seq_loss = self.fn(input[i, :seq_len, :seq_len, :seq_len, :seq_len],
                                   label[i, :seq_len, :seq_len, :seq_len, :seq_len],
                                   **fn_kwargs)
            else:
                logger.critical('too many dimensions for y_model, unsupported!')

            if loss_sqrt:
                loss_for_backprop += mi.sqrt(seq_loss)
                loss_vs_seq[i] = np.sqrt(seq_loss.numpy())
                # std_vs_seq[i] = np.sqrt(seq_loss.numpy().std())
            else:
                loss_for_backprop += seq_loss
                loss_vs_seq[i] = seq_loss.numpy()
                # std_vs_seq[i] = seq_loss.numpy().std()

        loss_for_backprop = 1.0 - loss_for_backprop
        loss_vs_seq = 1.0 - loss_vs_seq
        # logger.info(f'f1 score: {loss_for_backprop.numpy()}')

        return loss_for_backprop, loss_vs_seq, std_vs_seq


class SeqLossFn_P2P(nn.Layer):
    """ Returns a scalar loss, loss_vs_seq: [N], std_vs_seq: [N].
    Designed for functions that can calculate loss with padded zeros.
    One exception is f_score as padding affects both fp and tn counts
    std_vs_seq in only available if loss functions calculate losses point to point along "L"
    """

    def __init__(self, fn=F.mse_loss, name='mse', **kwargs):
        super(SeqLossFn_P2P, self).__init__()

        self.name = name.lower()
        self.fn = fn
        self.kwargs = kwargs

    def as_label(self, input):
        """ convert input to the same form as label, as model outputs may need
        to go through sigmod, softmax, etc.
        """
        if not isinstance(input, mi.Tensor):
            input = mi.to_tensor(input)

        if self.name in ['mse', 'bce']:
            pass
                    
        elif self.name in ['sigmoid+mse']:
            input = F.sigmoid(input)

        elif self.name in ['softmax+mse', 'softmax+bce']:
            input = mi.unstack(F.softmax(input, axis=-1), axis=-1)
            input = input[self.kwargs['label_col']]

        elif self.name in ['ce', 'crossentropy']:
            logger.critical('Need test! Help is the same as softmax+ce')

        elif self.name in ['softmax+ce', 'softmax+crossentropy']:
            if self.kwargs['soft_label']:
                input = F.softmax(input)
            else:
                input = F.softmax(input)
                input = mi.argmax(input, axis=-1)
        else:
            logger.critical(f'Cannot recognize loss_fn name: {self.name}!')

        return input

    def forward(self, input, label, seqs_len=None, loss_padding=False,
                loss_sqrt=False, **kwargs):
        """ return the loss with the same shape as input """

        batch_size = input.shape[0]

        fn_kwargs = self.kwargs.copy()
        fn_kwargs.update(kwargs)

        # deal with the specific requirements of the loss functions
        if self.name in ['softmax+crossentropy', 'crossentropy']:

            if self.kwargs.get('soft_label', None): # soft label
                if input.ndim > label.ndim:
                    label = hard2soft_label(label, nlabel=input.shape[-1], np=mi)
            else:  # hard label    
                if not isinstance(label, mi.Tensor) or label.dtype.name != 'INT64':
                    label = mi.to_tensor(label, dtype='int64')
                
                if input.ndim > label.ndim:
                    label = mi.unsqueeze(label, axis=-1)

        elif self.name in ['mse', 'sigmoid+mse']:
            if input.ndim == label.ndim + 1 and input.shape[-1] == 1:
                input = input.squeeze(-1)
                
        logger.debug(f'     num_dims: {input.ndim}')
        logger.debug(f'    data size: {batch_size}')
        logger.debug(f'     seqs_len: {seqs_len is not None}')
        logger.debug(f' loss_padding: {loss_padding}')
        logger.debug(f'    loss_sqrt: {loss_sqrt}')
        logger.debug(f'  loss kwargs: {fn_kwargs}')

            # if not isinstance(y_truth, mi.Tensor) or y_truth.dtype.name != 'FP32':
            #     y_truth = mi.to_tensor(y_truth, dtype='float32')

        # calculate all anyway, maybe more efficient for GPU
        loss_mat = self.fn(input, label, **fn_kwargs)

        if loss_padding: # process loss_mat as a whole
            if loss_mat.ndim == 1:
                # the std of the errors for each instance
                std_vs_seq = np.zeros_like(loss_mat, dtype=np.float32)
                # may need to squeeze the loss_mat
                loss_vs_seq = mi.squeeze(loss_mat, -1)
            else:
                # the axes for each instance, from the 2nd to the last
                inst_axes = tuple(range(1, loss_mat.ndim))
                std_vs_seq = loss_mat.numpy().std(axis=inst_axes)
                loss_vs_seq = mi.mean(loss_mat, axis=inst_axes)

            if loss_sqrt:
                loss_vs_seq = mi.sqrt(loss_vs_seq)
                std_vs_seq = np.sqrt(std_vs_seq)

            loss_for_backprop = mi.sum(loss_vs_seq)
            loss_vs_seq = loss_vs_seq.numpy()

        else: # deal each instance/sequence separately
            loss_for_backprop = mi.to_tensor(0.0, dtype='float32', stop_gradient=False)
            # loss_vs_seq = mi.zeros((batch_size,), dtype='float32')
            loss_vs_seq = np.zeros((batch_size), dtype=np.float32)
            std_vs_seq = np.zeros((batch_size), dtype=np.float32)
            for i in range(batch_size):
                seq_len = int(seqs_len[i])
                if loss_mat.ndim == 1:
                    seq_loss = loss_mat[i] # self.loss_fn(input[i], label[i], **kwargs)
                elif loss_mat.ndim == 2:
                    seq_loss = loss_mat[i, :seq_len]
                elif loss_mat.ndim == 3:
                    seq_loss = loss_mat[i, :seq_len, :seq_len]
                elif loss_mat.ndim == 4:
                    seq_loss = loss_mat[i, :seq_len, :seq_len, :seq_len]
                elif loss_mat.ndim == 5:
                    seq_loss = loss_mat[i, :seq_len, :seq_len, :seq_len, :seq_len]
                else:
                    logger.critical('too many dimensions for y_model, unsupported!')

                if loss_sqrt:
                    loss_for_backprop += mi.sqrt(mi.mean(seq_loss))
                    loss_vs_seq[i] = np.sqrt(seq_loss.numpy().mean())
                    std_vs_seq[i] = np.sqrt(seq_loss.numpy().std())
                else:
                    loss_for_backprop += mi.mean(seq_loss)
                    loss_vs_seq[i] = seq_loss.numpy().mean()
                    std_vs_seq[i] = seq_loss.numpy().std()

        # calculate the next loss_fn as needed
        # if self.loss_fn_next is not None:
        #     loss_for_backprop2, loss_vs_seq2, std_vs_seq2 = self.loss_fn_next(input, label,
        #                 seqs_len=seqs_len, loss_padding=loss_padding, loss_sqrt=loss_sqrt, **self.loss_fn_next_kwargs)

        #     loss_for_backprop += loss_for_backprop2
        #     loss_vs_seq += loss_vs_seq2
        #     std_vs_seq += std_vs_seq2

        return loss_for_backprop, loss_vs_seq, std_vs_seq


def get_loss_fn(args):
    """  """
    if isinstance(args.loss_fn, str):
        args.loss_fn = [args.loss_fn]
    args.loss_fn = [_s.lower() for _s in args.loss_fn]

    loss_fn = []
    logger.info(f'Getting loss function: {args.loss_fn}')

    if 'mse' in args.loss_fn:
        loss_fn.append(SeqLossFn_P2P(F.mse_loss, name='mse',
                reduction='none'))
                
    if 'bce' in args.loss_fn:
        loss_fn.append(SeqLossFn_P2P(F.binary_cross_entropy, name='bce',
                reduction='none'))

    if 'sigmoid+mse' in args.loss_fn:
        loss_fn.append(SeqLossFn_P2P(sigmoid_mse, name='sigmoid+mse',
                reduction='none'))

    if 'softmax+mse' in args.loss_fn:
        loss_fn.append(SeqLossFn_P2P(softmax_mse, name='softmax+mse',
                label_col=1, reduction='none'))
                
    if 'softmax+bce' in args.loss_fn:
        loss_fn.append(SeqLossFn_P2P(softmax_bce, name='softmax+bce',
                label_col=1, reduction='none'))
                
    if 'crossentropy' in args.loss_fn or 'ce' in args.loss_fn:
        loss_fn.append(SeqLossFn_P2P(F.cross_entropy, name='crossentropy', 
                soft_label=args.label_tone.lower() == 'soft'))
                
    if 'softmax+crossentropy' in args.loss_fn or 'softmax+ce' in args.loss_fn:
        loss_fn.append(SeqLossFn_P2P(F.softmax_with_cross_entropy, name='softmax+crossentropy',
                soft_label=args.label_tone.lower() == 'soft'))
                
    if 'fscore' in args.loss_fn or 'f-score' in args.loss_fn:
        loss_fn.append(SeqLossFn_Agg(f_score, name='fscore',
                label_col=1, batch=False, beta=1.0, epsilon=1e-7))

    if len(loss_fn) == 0:
        logger.critical(f'not supported loss functions found in: {args.loss_fn}!')

    return loss_fn


def save_loss_csv(save_file, loss_df, groupby=None):
    """  """
    df = loss_df
    if groupby is not None:
        col = [_col for _col in groupby if _col in loss_df.columns]
        if col:
            logger.info(f'Grouping data by: {col} before saving')
            df = loss_df.groupby(col).mean().reset_index()

    if df is not None:
        df.to_csv(save_file, index=False, float_format='%.4g')
    # np.savetxt(save_file, train_loss, fmt='%6d ' + '%8.4f ' * 4 + ' %5d'*4)

def save_model_prediction(y_model, save_dir='./', seqs_len=None, istart=1, stem='predict'):
    """  """
    num_seqs = len(y_model)
    save_dir = path(save_dir)

    ndim = y_model[0].ndim
    for i in range(num_seqs):
        if seqs_len is None:
            y_save = y_model[i]
        else:
            seq_len = int(seqs_len[i])
            if ndim == 0:
                y_save = y_model[i]
            elif ndim == 1:
                y_save = y_model[i][:seq_len]
            elif ndim == 2:
                y_save = y_model[i][:seq_len, :seq_len]
            elif ndim == 3:
                y_save = y_model[i][:seq_len, :seq_len, :seq_len]
            elif ndim == 4:
                y_save = y_model[i][:seq_len, :seq_len, :seq_len, :seq_len]
            else:
                logger.critical('too many dimensions to save')
            
        if ndim <= 2:
            np.savetxt(save_dir / f'{istart + i}.{stem}.txt', y_save, fmt='%10.8f')
        else:
            np.save(save_dir / f'{istart + i}.{stem}', y_save)


def compute_loss(loss_fn, input, label, seqs_len=None, shuffle=False, batch_size=23, **kwargs):
    """ both input and label can be list/array/tensor
    But the first dimension must be the batch_size
    """
    if type(loss_fn) not in (list, tuple): loss_fn = [loss_fn]
    num_data = len(input)
    if seqs_len is None:
        midata = list(zip(input, label))
    else:
        midata = list(zip(input, seqs_len, label))

    miloader = get_dataloader(midata, batch_size=batch_size, shuffle=shuffle)

    loss_vs_seq = np.zeros((num_data), dtype=np.float32)
    std_vs_seq = np.zeros((num_data), dtype=np.float32)

    for ibatch, data in enumerate(miloader()):
        num_seqs = data[0].shape[0]
        istart = ibatch * batch_size
        iend = istart + num_seqs
        if seqs_len is not None:
            seqlen_batch = data[1]
        else:
            seqlen_batch = None

        for one_loss_fn in loss_fn:
            _, _loss_vs_seq, _std_vs_seq = one_loss_fn(data[0], data[-1],
                        seqs_len=seqlen_batch, **kwargs)
            loss_vs_seq[istart:iend] = loss_vs_seq[istart:iend] + _loss_vs_seq
            std_vs_seq[istart:iend] = std_vs_seq[istart:iend] + _std_vs_seq

    return loss_vs_seq, std_vs_seq


def train(model, midata, **kwargs):
    """ Bad practices:
        1) args procesing is odd (kwargs > model.args > args )
        2) fields are added to model structure
    """
    # default settings
    args = misc.Struct(dict(
                       trainloss_patience = 5, trainloss_rdiff = 1e-3,
                       validloss_patience = 3, validloss_rdiff = 1e-3,
                       validate_callback = None,
                       num_callbacks_per_epoch = 10,
                       lr_scheduler = 'none',
                       num_recaps_per_epoch = 30,
                       num_epochs = 2,
                       batch_size = 2,
                       loss_padding = False,
                       shuffle = True,
                       save_dir = None,
                       save_level = 1,
                       verbose = 1,
                       ))
    args.update(vars(model.args)) # model.args overwrite default args
    args.update(kwargs) # kwargs overwrite all
    if isinstance(args.save_dir, str): args.save_dir = path(args.save_dir)
    if args.save_dir: args.save_dir.mkdir(parents=True, exist_ok=True)
    model.args.update(vars(args)) # args should not change anymore

    # XQ: do not need this yet, save it for later
    # num_workers = mi.distributed.ParallelEnv().nranks
    # work_site = mi.CUDAPlace(mi.distributed.ParallelEnv().dev_id) if num_workers > 1 else mi.CUDAPlace(0)
    # exe = mi.Executor(work_site)

    if args.data_size > 0:
        if args.data_size < len(midata):
            midata = random_sample(midata, size=args.data_size, replace=False)
        elif args.data_size == len(midata):
            logger.warning(f'Specified data size: {args.data_size} == data length: {len(midata)}.')
        else:
            logger.warning(f'Specified data size: {args.data_size} > data length: {len(midata)}!')

    miloader = get_dataloader(midata, batch_size=args.batch_size, shuffle=args.shuffle)
    model.num_batches = len(miloader)
    model.num_data = len(midata)

    # model.train_loss = np.zeros((model.num_data * args.num_epochs, 7), dtype=np.float32)
    # model.train_loss = np.zeros((my_args.num_epochs * model.num_batches, 9), dtype=np.float)
    model.train_loss = [] # a list of DataFrames (concatenated at the end)
    validate_hist = misc.Struct(valid_loss=[]) # consider to include this in the model structure?
    model.validate_hist  = validate_hist

    callback_interval = max([1, model.num_batches // args.num_callbacks_per_epoch])
    recap_interval = model.num_batches // args.num_recaps_per_epoch + 1
    # num_recaps = int(np.ceil(model.num_batches / recap_interval))
    logger.info(f'Training, data size: {len(midata)}')
    logger.info(f'         batch size: {args.batch_size}')
    logger.info(f'            shuffle: {args.shuffle}')
    logger.info(f'       # of batches: {model.num_batches}')
    logger.info(f'     recap interval: {recap_interval}')
    logger.info(f'  validate interval: {callback_interval}')
    logger.info(f'        # of epochs: {args.num_epochs}')
    logger.info(f'       loss padding: {args.loss_padding}')

    # temporary vars for journaling, not saved to files
    loss_vs_epoch = pd.DataFrame() # np.array([], dtype=np.float32)
    # loss_vs_batch = pd.DataFrame() # do not need this yet
    loss_for_recap, std_for_recap = [], [] # accumulate results between recaps

    model.net.train()
    for model.epoch in range(args.num_epochs):
        loss_one_epoch = np.empty((model.num_data, 7), dtype=np.float32)
        for model.batch, data in enumerate(miloader()):
            model.optim.clear_grad()

            # data: [seq_in, upp_truth, [seq_len, idx]]
            x, y_truth = data[0], data[-1]
            seqs_len, seqs_idx = data[1][:, 0], data[1][:, 1]

            # if not isinstance(y_truth, mi.Tensor) or y_truth.dtype.name != 'FP32':
            #     y_truth = mi.to_tensor(y_truth, dtype='float32')
            # if not isinstance(seqs_len, mi.Tensor) or seqs_len.dtype.name != 'INT32':
                # seqs_len = mi.to_tensor(seqs_len, dtype='int32')

            if x.ndim > 1 and x.shape[0] == 1 and x.shape[1] > seqs_len[0] and not args.loss_padding:
                x = cut_padding(x, int(seqs_len[0]))
                y_truth = cut_padding(y_truth, int(seqs_len[0]))

            y_model = model.net(x, seqs_len)

            num_seqs = y_model.shape[0]
            seqs_len, seqs_idx = seqs_len.numpy(), seqs_idx.numpy()

            # if args.loss_padding: # inlucde loss from padded sequences
            #     loss_for_backprop = model.loss_fn(y_model, y_truth, reduction='none')
            #     # F.mse_loss(y_model, y_truth, reduction='none')
            #     std_vs_seq = np.sqrt(loss_for_backprop.numpy().std(axis=-1)) # for recap and log

            #     loss_for_backprop = mi.sqrt(mi.mean(loss_for_backprop, axis=-1))
            #     loss_vs_seq = loss_for_backprop.numpy() # for recap and log

            #     loss_for_backprop = loss_for_backprop.sum()
            # else:
            #     loss_for_backprop = mi.to_tensor(0.0, dtype='float32')
            #     loss_vs_seq = np.zeros((num_seqs, ), dtype=np.float32)
            #     std_vs_seq = np.zeros((num_seqs, ), dtype=np.float32)

            #     for i in range(num_seqs):
            #         seq_len = int(seqs_len[i])
            #         if len(y_model.shape) == 1:
            #             seq_loss = model.loss_fn(y_model[i], y_truth[i])
            #         elif len(y_model.shape) == 2:
            #             seq_loss = model.loss_fn(y_model[i, :seq_len], y_truth[i, :seq_len])
            #         elif len(y_model.shape) == 3:
            #             seq_loss = model.loss_fn(y_model[i, :seq_len, :seq_len],
            #                                      y_truth[i, :seq_len, :seq_len])
            #         elif len(y_model.shape) == 4:
            #             seq_loss = model.loss_fn(y_model[i, :seq_len, :seq_len, :seq_len],
            #                                      y_truth[i, :seq_len, :seq_len, :seq_len])
            #         else:
            #             logger.critical('too many dimensions for y_model, unsupported!')

            #         # F.mse_loss(y_model[i, :seq_len], y_truth[i, :seq_len], reduction='none')
            #         loss_for_backprop += mi.sqrt(mi.mean(seq_loss))

            #         numpy_tmp = seq_loss.numpy()
            #         std_vs_seq[i] = np.sqrt(numpy_tmp.std())
            #         loss_vs_seq[i] = np.sqrt(numpy_tmp.mean())

            # acc = mi.metric.accuracy(mi.flatten(y_model), mi.flatten(y_truth))
            loss_for_backprop = mi.to_tensor(0.0, dtype='float32', stop_gradient=False)
            loss_vs_seq = np.zeros((num_seqs), dtype=np.float32)
            std_vs_seq = np.zeros((num_seqs), dtype=np.float32)
            for loss_fn in model.loss_fn:
                _loss_for_backprop, _loss_vs_seq, _std_vs_seq = loss_fn(y_model, y_truth,
                            seqs_len=seqs_len, loss_padding=args.loss_padding,
                            loss_sqrt=args.loss_sqrt)
                loss_for_backprop += _loss_for_backprop
                loss_vs_seq += _loss_vs_seq
                std_vs_seq += _std_vs_seq

            # loss = loss_for_backprop.mean() # + 1 / ( 1 + 2 * mi.square(y_model - 0.5).mean())
            loss_for_backprop.backward() # loss is loss_per_batch by convention
            if args.verbose > 1:
                print("Current state of net parameters:")
                for par_n, par_v in model.net.named_parameters():

                    print(f'{par_n:28s} - min: {par_v.min().numpy()[0]:11.6f}, max: {par_v.max().numpy()[0]:11.6f}, ' + \
                            f'grad_min: {par_v.grad.min().item():11.6f}, grad_max: {par_v.grad.max().item():11.6f}')

            # save&display progress
            ibatch = model.epoch * model.num_batches + model.batch
            istart = model.batch * args.batch_size # model.epoch * model.num_data +
            iend = istart + num_seqs
            loss_one_epoch[istart:iend, 0] = ibatch
            loss_one_epoch[istart:iend, 1] = loss_vs_seq
            loss_one_epoch[istart:iend, 2] = std_vs_seq
            loss_one_epoch[istart:iend, 3] = seqs_len
            loss_one_epoch[istart:iend, 4] = seqs_idx
            loss_one_epoch[istart:iend, 5] = model.epoch
            loss_one_epoch[istart:iend, 6] = model.batch

            loss_for_recap.extend(loss_vs_seq)
            std_for_recap.extend(std_vs_seq)

            if model.batch % recap_interval == 0: # recap the 0th!
                loss_for_recap = np.array(loss_for_recap).mean()
                std_for_recap = np.array(std_for_recap).mean()
                logger.info(f'Epoch/batch: {model.epoch:d}/{model.batch:4d}, ibatch: {ibatch:4d}, ' + \
                            f'loss: \033[0;36m{loss_for_recap:6.4f}\033[0m, std: {std_for_recap:6.4f}')

                # learning_rate tuning
                if not isinstance(model.lr_scheduler, float): # mi.optimizer.lr.LRScheduler):
                    model.lr_scheduler.step(loss_for_recap)

                loss_for_recap, std_for_recap = [], []

            # callback (usually for early stopping)
            if args.validate_callback is not None and (model.batch % callback_interval == 0
                        or model.batch == model.num_batches - 1):
                validate_hist.update(epoch=model.epoch, batch=model.batch, ibatch=ibatch)
                validate_hist = args.validate_callback(model=model, history=validate_hist)
                model.net.train()

            model.optim.step()

        # post-epoch
        loss_one_epoch = pd.DataFrame(loss_one_epoch, columns=['ibatch', 'loss', 'loss_std', 'seq_len', 'idx','epoch', 'batch'])
        model.train_loss.append(loss_one_epoch) # this will be saved as csv

        # loss_vs_epoch = np.concatenate((loss_vs_epoch, loss_one_epoch.mean(axis=0, keepdims=True)), axis=0)
        # loss_vs_epoch.append(model.train_loss[istart:iend, 1].mean())

        # Need to check whether pd.append and pd.grouby retains the order

        loss_vs_epoch = loss_vs_epoch.append(loss_one_epoch.mean(), ignore_index=True)
        # loss_vs_batch = loss_vs_batch.append(loss_one_epoch.groupby('batch').mean(), ignore_index=True)

        logger.info(f'Epoch {model.epoch} average training loss: ' +
                    f'\033[0;46m{loss_vs_epoch.loss.iat[-1]:6.4f}\033[0m' +
                    f' std: {loss_vs_epoch.loss_std.iat[-1]:6.4f}')

        valid_vs_epoch = validate_hist.loss_per_call.groupby('epoch').mean().reset_index()
        logger.info(f'Epoch {model.epoch} average validate loss: ' +
                    f'\033[0;46m{valid_vs_epoch.loss.iat[-1]:6.4f}\033[0m' +
                    f' std: {valid_vs_epoch.loss_std.iat[-1]:6.4f}')

        if args.save_dir and args.save_level >= 2:
            epoch_save_dir = args.save_dir / 'epoch_log'
            epoch_save_dir.mkdir(parents=True, exist_ok=True)
            save_loss_csv(epoch_save_dir / f'train_epo{model.epoch:03d}_{loss_vs_epoch.loss.iat[-1]:6.4f}.csv', loss_one_epoch)
            save_loss_csv(epoch_save_dir / f'valid_epo{model.epoch:03d}_{valid_vs_epoch.loss.iat[-1]:6.4f}.csv', validate_hist.valid_loss[-1])

        # stop the train if needed
        if model.epoch >= args.trainloss_patience and model.epoch >= args.validloss_patience:

            if all((loss_vs_epoch.loss.diff() > 0)[-args.trainloss_patience:]):
                logger.warning(f'Training loss increased {args.trainloss_patience} consecutive epochs, stopping!!!')
                break
            if all(loss_vs_epoch.loss.pct_change().abs()[-args.trainloss_patience:] < args.trainloss_rdiff):
                logger.warning(f'Training loss changed < {args.trainloss_rdiff} for {args.trainloss_patience} consecutive epochs, stopping!!!')
                break
            if all((valid_vs_epoch.loss.diff() > 0)[-args.validloss_patience:]):
                logger.warning(f'Validate loss increased {args.validloss_patience} consecutive epochs, stopping!!!')
                break
            if all(valid_vs_epoch.loss.pct_change().abs()[-args.validloss_patience:] < args.validloss_rdiff):
                logger.warning(f'Validate loss changed < {args.validloss_rdiff} for {args.validloss_patience} consecutive epochs, stopping!!!')
                break

    # post-train
    model.train_loss = pd.concat(model.train_loss)
    model.train_loss_vs_epoch = loss_vs_epoch

    model.valid_loss = pd.concat(validate_hist.valid_loss)
    model.valid_loss_vs_epoch = validate_hist.loss_per_call.groupby('epoch').min().reset_index()

    model.validate_hist  = validate_hist

    if args.save_dir and args.save_level >= 1: # save the final model (maybe unnecessary)
        logger.info(f'Saving final results in <{args.save_dir}>...')
        state_dict_save(model, fdir=args.save_dir)
        args.net_src_file = save_net_pycode(args.net_src_file, args.save_dir)
        save_loss_csv(args.save_dir / 'train_log.csv', model.train_loss, groupby=args.save_grpby)
        save_loss_csv(args.save_dir / 'valid_log.csv', model.valid_loss, groupby=args.save_grpby)

    return model.train_loss, validate_hist.valid_loss


def validate(model, midata, **kwargs):
    """ model structure is not changed during this call """
    args = misc.Struct(dict(batch_size = 512,
                            shuffle = False,
                            num_recaps_per_epoch = 10,
                            save_dir = None,
                            verbose = 1,
                            ))
    args.update(vars(model.args))
    args.update(kwargs) # kwargs rule all
    if isinstance(args.save_dir, str): args.save_dir = path(args.save_dir)
    model.args.update(vars(args)) # args should not change anymore

    miloader = get_dataloader(midata, batch_size=args.batch_size, shuffle=args.shuffle)
    # return: [ibatch, rmsd, std, seq_len, idx (in the original seq data)]
    valid_loss = np.zeros((len(midata), 5), dtype=np.float32)

    recap_interval = len(miloader) // args.num_recaps_per_epoch + 1
    logger.info(f'Validating, data size: {len(midata)}')
    logger.info(f'           batch size: {args.batch_size}')
    logger.info(f'              shuffle: {args.shuffle}')
    logger.info(f'         # of batches: {len(miloader)}')
    logger.info(f'       recap interval: {recap_interval}')
    logger.info(f'         loss padding: {args.loss_padding}')

    model.net.eval()
    with mi.no_grad():
        loss_for_recap, std_for_recap = [], []
        for ibatch, data in enumerate(miloader()):
            # data: [seq_in, upp_truth, [seq_len, idx]]
            x, y_truth,= data[0], data[-1]
            seqs_len, seqs_idx = data[1][:,0], data[1][:,1]

            # if not isinstance(y_truth, mi.Tensor) or y_truth.dtype.name != 'FP32':
            #     y_truth = mi.to_tensor(y_truth, dtype='float32')

            if x.ndim > 1 and x.shape[0] == 1 and x.shape[1] > seqs_len[0] and not args.loss_padding:
                x = cut_padding(x, int(seqs_len[0]))
                y_truth = cut_padding(y_truth, int(seqs_len[0]))

            y_model = model.net(x, seqs_len)

            num_seqs = y_model.shape[0]
            seqs_len, seqs_idx = seqs_len.numpy(), seqs_idx.numpy()

            # loss_vs_seq = np.zeros((num_seqs,), dtype=np.float32)
            # std_vs_seq = np.zeros((num_seqs,), dtype=np.float32)

            # for i in range(num_seqs):
            #     seq_len = int(seqs_len[i])
            #     seq_loss = F.mse_loss(y_model[i, :seq_len], y_truth[i, :seq_len], reduction='none').numpy()
            #     loss_vs_seq[i] = np.sqrt(seq_loss.mean())
            #     std_vs_seq[i] = np.sqrt(seq_loss).std()

            loss_vs_seq = np.zeros((num_seqs), dtype=np.float32)
            std_vs_seq = np.zeros((num_seqs), dtype=np.float32)
            for loss_fn in model.loss_fn:
                _, _loss_vs_seq, _std_vs_seq = loss_fn(y_model, y_truth,
                            seqs_len=seqs_len, loss_padding=args.loss_padding, loss_sqrt=args.loss_sqrt)
                loss_vs_seq += _loss_vs_seq
                std_vs_seq += _std_vs_seq

            istart = ibatch * args.batch_size
            iend = istart + num_seqs
            valid_loss[istart:iend, 0] = ibatch
            valid_loss[istart:iend, 1] = loss_vs_seq
            valid_loss[istart:iend, 2] = std_vs_seq
            valid_loss[istart:iend, 3] = seqs_len
            valid_loss[istart:iend, 4] = seqs_idx

            loss_for_recap.extend(loss_vs_seq)
            std_for_recap.extend(std_vs_seq)
            if ibatch % recap_interval == 0:
                loss_for_recap = np.array(loss_for_recap).mean()
                std_for_recap = np.array(std_for_recap).mean()
                logger.info(f'ibatch: {ibatch:4d}, loss: {loss_for_recap:6.4f}, std: {std_for_recap:6.4f}')
                loss_for_recap, std_for_recap = [], []

    valid_loss = pd.DataFrame(valid_loss, columns=['ibatch', 'loss', 'loss_std', 'seq_len', 'idx'])
    logger.info(f'Validate mean: \033[0;46m{valid_loss.loss.mean():6.4f}\033[0m' +
                f', std: {valid_loss.loss.std():6.4f}')

    if args.save_dir and args.save_level >= 1:
        if not args.save_dir.exists(): args.save_dir.mkdir(parents=True)
        save_loss_csv(args.save_dir / 'valid_log.csv', valid_loss)
    return valid_loss


def predict(model, midata, **kwargs):
    """  """
    args = misc.Struct(dict(
                    batch_size = 128,
                    shuffle = False, # the first two not used yet
                    num_recaps_per_epoch = 10,
                    save_dir = path.cwd() / 'predict.files',
                    ))
    args.update(vars(model.args))
    args.update(kwargs) # kwargs rule all
    if args.save_dir and not isinstance(args.save_dir, path):
        args.save_dir = path(args.save_dir)
    model.args.update(vars(args))

    miloader = get_dataloader(midata, batch_size=args.batch_size, shuffle=args.shuffle)

    recap_interval = len(miloader) // args.num_recaps_per_epoch + 1
    data_size = len(midata)

    logger.info(f'Predicting, data size: {data_size}')
    logger.info(f'           batch size: {args.batch_size}')
    logger.info(f'              shuffle: {args.shuffle}')
    logger.info(f'         # of batches: {len(miloader)}')
    logger.info(f'       recap interval: {recap_interval}')

    if args.save_dir and args.save_level >= 1:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Predicted files will be saved in: {args.save_dir}')

    # two returned values
    y_model_all = [] # np.empty((data_size, midata[0][0].shape[0]), dtype=np.float32)
    seqs_len_all = np.empty((data_size), dtype=np.int32)

    model.net.eval()
    with mi.no_grad(), tqdm(total=data_size, disable=False) as prog_bar:
        for ibatch, data in enumerate(miloader()):
            x, y_truth = data[0], data[-1]
            seqs_len, seqs_idx = data[1][:, 0], data[1][:, 1]

            if x.ndim > 1 and x.shape[0] == 1 and x.shape[1] > seqs_len[0] and not args.loss_padding:
                x = cut_padding(x, int(seqs_len[0]))
                y_truth = cut_padding(y_truth, int(seqs_len[0]))

            # y_model = upp_model.net(mi.unsqueeze(x, 0)) # add the batch_size dim
            y_model = model.net(x, seqs_len)
            num_seqs = y_model.shape[0]

            istart = ibatch * args.batch_size
            iend = istart + num_seqs
            # y_model_all[istart:iend, :] = y_model.numpy()
            y_model_all.extend(y_model.numpy())
            seqs_len_all[istart:iend] = seqs_len.numpy()

            prog_bar.update(num_seqs)
            if args.save_dir:
                # in the case of multiple loss_fns, their as_label() should give the same results
                # for loss_fn in model.loss_fn:
                y_model = model.loss_fn[0].as_label(y_model)
                save_model_prediction(y_model.numpy(), args.save_dir, seqs_len=seqs_len,
                            istart=ibatch * args.batch_size + 1, stem='predict')

                # for i in range(num_seqs):
                #     seq_len = int(seqs_len[i])
                #     if y_model.ndim == 1:
                #         y_save = y_model[i]
                #     elif y_model.ndim == 2:
                #         y_save = y_model[i, :seq_len]
                #     elif y_model.ndim == 3:
                #         y_save = y_model[i, :seq_len, :seq_len]
                #     elif y_model.ndim == 4:
                #         y_save = y_model[i, :seq_len, :seq_len, :seq_len]
                #     else:
                #         logger.critical('too many dimensions to save')
                    
                #     if y_model.ndim < 4:
                #         np.savetxt(args.save_dir / f'{ibatch * args.batch_size + i + 1}.predict.txt',
                #                     y_save.numpy(), fmt='%6.4f')
                #     else:
                #         np.save(args.save_dir / f'{ibatch * args.batch_size + i + 1}.predict', y_save.numpy())

            # if ibatch % recap_interval == 0:
                # logger.info(f'{ibatch} out of {data_size} ({int(ibatch / data_size * 100)}%) done')

    logger.info(f'Completed prediction of {data_size} samples')
    return y_model_all, seqs_len_all


def state_dict_save(upp_model, fdir=path.cwd()):
    """  """
    if isinstance(fdir, str): fdir = path(fdir)
    if not fdir.exists(): fdir.mkdir(parents=True)

    net_state_file = fdir / 'net.state'
    opt_state_file = fdir / 'opt.state'

    mi.save(upp_model.net.state_dict(), net_state_file)
    mi.save(upp_model.optim.state_dict(), opt_state_file)

    # mi.jit.save(model.net, (fdir / 'model').as_posix())
    logger.info(f'Saved model states in: {fdir}')


def state_dict_load(upp_model, fdir=path.cwd()):

    if isinstance(fdir, str): fdir = path(fdir)
    logger.info(f'Loading model states from: {fdir}')

    net_state_file = fdir / 'net.state'
    opt_state_file = fdir / 'opt.state'

    try:
        if net_state_file.exists():
            net_state_dict = mi.load(net_state_file)
            upp_model.net.set_state_dict(net_state_dict)
            logger.info(f'Loaded net state: {net_state_file}')
    except:
        logger.warning('Error in net state_dict loading!')

    try:
        if opt_state_file.exists():
            opt_state_dict = mi.load(opt_state_file)
            upp_model.optim.set_state_dict(opt_state_dict, use_structured_name=False)
            logger.info(f'Loaded optim state: {opt_state_file}')
    except:
        logger.warning('Error in optim state_dict loading!')


def get_model(args, quiet=False):
    """  """
    # model = mi.Model(upp_net)
    # model.prepare(upp_opt, loss_fn)
    # model.fit(train_dataset, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, log_freq=200)
    # model.evaluate(test_data, log_freq=20, batch_size=BATCH_SIZE)
    upp_model = misc.Struct()
    upp_model.args = args
    upp_model.net = get_net(args, quiet=quiet)
    upp_model.optim, upp_model.lr_scheduler = get_optimizer(upp_model.net, args)
    upp_model.loss_fn = get_loss_fn(args)
    return upp_model


def validate_in_train(model=None, midata=None, history=misc.Struct(), **kwargs):
    """  """
    configs = misc.Struct(
        epoch = 0, # the following three are from the train()
        batch = 0, # so as to align with the training curve
        ibatch = 0,
        batch_size = 128, # for early stop only
        times_called = 0,
        # valid_loss = pd.DataFrame(),
        valid_loss = [],
        loss_per_call = pd.DataFrame(columns=['ibatch', 'loss', 'loss_std', 'epoch', 'batch']),
        save_dir = None,
        saved_idx = [], # this points to loss_hist
        saved_dirs = [],
        valid_loss_best = 1000.0,
        keep_best_only = True,
        verbose = 1,
    )
    configs.update(vars(history))
    configs.update(kwargs)
    configs.times_called += 1

    # this is different from args.save_dir
    if configs.save_dir and isinstance(configs.save_dir, str):
        configs.save_dir = path(configs.save_dir)

    misc.logger_setlevel(logger, 0)
    valid_loss = validate(model, midata, save_dir=None, shuffle=False, batch_size=configs.batch_size)
    misc.logger_setlevel(logger, configs.verbose)

    valid_loss_avg = valid_loss.loss.mean()
    valid_loss_std = valid_loss.loss_std.mean()
    logger.info(f'loss: \033[0;32m{valid_loss_avg:6.4f}\033[0m, std: {valid_loss_std:6.4f}')

    # configs.valid_loss = configs.valid_loss.append(valid_loss.assign(ibatch=configs.ibatch,
    #             epoch=configs.epoch, batch=configs.batch), ignore_index=True)
    configs.valid_loss.append(valid_loss.assign(ibatch=configs.ibatch,
                epoch=configs.epoch, batch=configs.batch))

    configs.loss_per_call = configs.loss_per_call.append(dict(ibatch=configs.ibatch,
                loss=valid_loss_avg, loss_std=valid_loss_std,
                epoch=configs.epoch, batch=configs.batch), ignore_index=True)

    if configs.save_dir and valid_loss_avg < configs.valid_loss_best and configs.times_called > 1:
        configs.valid_loss_best = valid_loss_avg

        # call "next_backup_path" just in case it exists
        new_save_dir = gwio.next_backup_path(configs.save_dir / f'earlystop_{valid_loss_avg:6.4f}')

        state_dict_save(model, fdir=new_save_dir)
        model.args.net_src_file = save_net_pycode(model.args.net_src_file, new_save_dir)
        gwio.dict2json(vars(model.args), new_save_dir / 'args.json')
        save_loss_csv(new_save_dir / 'valid_loss.csv', valid_loss)
        logger.info(f'Saved best model: {new_save_dir}')

        configs.saved_dirs.append(new_save_dir)
        configs.saved_idx.append(configs.loss_per_call.shape[0] - 1)

        if configs.keep_best_only:
            for old_save_dir in configs.saved_dirs[-2:-1]:
                if not old_save_dir.exists() or new_save_dir.samefile(old_save_dir): continue
                logger.info(f'Removing earlystop model: {old_save_dir}')
                shutil.rmtree(old_save_dir)

    return configs
    # model.valid_rmsd[model.epoch*num_recaps + model.batch // update_interval, :] = \
        # [model.epoch*len(miloader)+model.batch, valid_rmsd_avg, model.epoch, model.batch]


def scan_data_args(args, midata, data_sizes=None, batch_sizes=None, **kwargs):
    """ data/batch_sizes='auto'|None|int|list/array """
    args.update(kwargs)

    if args.save_dir and isinstance(args.save_dir, str):
        args.save_dir = path(args.save_dir)
    if args.save_dir: args.save_dir.mkdir(parents=True, exist_ok=True)

    num_data = len(midata)

    # get data grids, the default goes up from 1 by a factor 2
    def get_data_grid(val_in, default=1):
        if val_in is None:
            val_out = [default]
        elif isinstance(val_in, str):
            if val_in.lower() == 'auto':
                num_grids = int(np.log2(num_data)) + 1
                val_out = np.logspace(0, num_grids - 1, num=num_grids, base=2, dtype=int)
            elif val_in.lower() == 'all': # all
                val_out = [num_data]
            else:
                val_out = [1, num_data]
        elif isinstance(val_in, int):
            val_out = [val_in]
        else:
            val_out = np.array(val_in, dtype=int)
        return val_out

    data_sizes = get_data_grid(data_sizes, default=num_data)
    batch_sizes = get_data_grid(batch_sizes, default=args.batch_size)

    scan_best_loss = pd.DataFrame() # np.zeros((len(data_sizes)*len(batch_sizes), 6), dtype=float)
    # the lists here store the train_loss, etc. from each scan
    scan_train_loss = []
    scan_valid_loss = [] # this is the callback return

    data_indices = np.linspace(0, num_data-1, num=num_data, dtype=int)
    for i, (data_size, batch_size) in enumerate(itertools.product(data_sizes, batch_sizes)):
        logger.info(f'scan #{i}, data_size: {data_size}, batch_size: {batch_size}')
        batch_size = int(batch_size) # somehow neede for paddle

        # get train data
        if 0 < data_size < num_data:
            data_indices = np.random.permutation(data_indices)
            train_data = [midata[data_indices[_i]] for _i in range(data_size)]
        else:
            train_data = midata

        # train with chosen data and batch_size
        logger.info('Creating a new model...') # or re-initialize the model
        model = get_model(args)

        # both returns are pd.DataFrames of
        save_dir = args.save_dir
        train(model, train_data, batch_size=batch_size, save_dir=None, validate_callback=args.validate_callback)
        args.save_dir = save_dir

        # should consier to reduce the level to batch, at least
        scan_train_loss.append(model.train_loss.assign(data_size=data_size, batch_size=batch_size))
        scan_valid_loss.append(model.valid_loss.assign(data_size=data_size, batch_size=batch_size))

        # np.tile([data_size, batch_size], (train_loss.shape[0], 1))
        # train_rmsd[-1,-2]: the last epoch
        # last_train_epoch = train_loss[train_loss.epoch == train_loss.epoch.iat[-1]]
        # last_estop_epoch = estop_loss[estop_loss.epoch == train_loss.epoch.iat[-1]]

        scan_best_loss = scan_best_loss.append(dict(
                    data_size = data_size,
                    batch_size = batch_size,
                    train_loss = model.train_loss_vs_epoch.loss.min(),
                    valid_loss = model.valid_loss_vs_epoch.loss.min(),
                    ), ignore_index=True)

        if args.save_dir: # save all the train and valid curves from each call
            save_prefix = f'scan_data_size{data_sizes[0]}-{data_sizes[-1]}' + \
                          f'_batch{batch_sizes[0]}-{batch_sizes[-1]}'

            save_file = args.save_dir / (save_prefix + '_train.csv')
            save_loss_csv(save_file, pd.concat(scan_train_loss), groupby=['data_size', 'batch_size', 'ibatch'])
            logger.info(f'Saved train curve: {save_file}')

            save_file = args.save_dir / (save_prefix + '_valid.csv')
            save_loss_csv(save_file, pd.concat(scan_valid_loss), groupby=['data_size', 'batch_size', 'ibatch'])
            logger.info(f'Saved valid curve: {save_file}')

            save_file = args.save_dir / (save_prefix + '_best.csv')
            scan_best_loss.to_csv(save_file, index=False, float_format='%.4g')
            logger.info(f'Saved scan summary: {save_file}')

        # valid_loss = validate(model, valid_data, args=args, save_dir=None, shuffle=False, batch_size=512)
        # scan_best_loss[i,[4,5]] = valid_loss.loc[:,['loss', 'loss_std']].mean(axis=0)
        # scan_valid_loss.append(valid_loss.assign(
        #     data_size = [data_size] * valid_loss.shape[0],
        #     batch_size = [batch_size] * valid_loss.shape[0]))

        # np.append(np.tile([data_size, batch_size], (valid_rmsd.shape[0], 1)),
        #     valid_rmsd, axis=1)

    return scan_best_loss


def scout_args(args, train_set, valid_set=None, arg_names=None, arg_values=None,
            grid_search=False, **kwargs):
    """ both arg_names and argvalues are lists of MATCHING names/values """
    # take care of args
    args.update(kwargs)

    if args.save_dir and isinstance(args.save_dir, str):
        args.save_dir = path(args.save_dir)
    if args.save_dir: args.save_dir.mkdir(parents=True, exist_ok=True)

    # scan_best_loss = np.zeros((np.prod(arg_lens), 6), dtype=float)
    scan_best_loss = pd.DataFrame(columns=arg_names + ['train_loss', 'valid_loss'], dtype=float)

    scan_train_loss = []
    scan_valid_loss = []

    if grid_search:
        arg_sets = list(itertools.product(*arg_values))
    else:
        arg_sets = list(zip(*arg_values))

    for i, value_set in enumerate(arg_sets):
        scan_args = dict(zip(arg_names, value_set))
        scan_best_loss.loc[i, arg_names] = value_set
        logger.info(f'args set #: {i}/{len(arg_sets)}, {scan_args}')

        args.update(scan_args)

        if args.rebake_midata: # midata should be the pkldata!!!
            args = autoconfig_args(args)
            train_data = bake_midata(train_set, args)
            if valid_set is not None:
                valid_data = bake_midata(valid_set, args)
                args.validate_callback = func_partial(validate_in_train, midata=valid_data,
                            save_dir=args.save_dir, verbose=args.verbose)
        else:
            train_data = train_set

        model = get_model(args)

        save_dir = args.save_dir # train() overwrites save_dir
        train(model, train_data, save_dir=None, validate_callback=args.validate_callback)
        args.save_dir = save_dir

        # reduce the train_loss???
        scan_train_loss.append(model.train_loss.assign(**scan_args)) # append scan_args
        scan_valid_loss.append(model.valid_loss.assign(**scan_args))

        scan_best_loss.loc[i, 'train_loss'] = model.train_loss_vs_epoch.loss.min()
        scan_best_loss.loc[i, 'valid_loss'] = model.valid_loss_vs_epoch.loss.min()

        # if valid_data is None: continue

        # valid_loss = validate(model, valid_data, args=args, save_dir=None, batch_size=512)
        # scan_best_loss.loc[i, ['valid_loss', 'valid_loss_std']] = valid_loss.loc[:,['loss', 'loss_std']].mean(axis=0)
        # scan_valid_loss.append(valid_loss.assign(**scan_args))

        # Saving results
        if args.save_dir: # save all the train and valid curves from each call
            save_prefix = 'scan_args_' + '-'.join(arg_names)

            save_file = args.save_dir / (save_prefix + '_train.csv')
            save_loss_csv(save_file, pd.concat(scan_train_loss), groupby=arg_names + ['ibatch'])
            logger.info(f'Saved train curve: {save_file}')

            save_file = args.save_dir / (save_prefix + '_valid.csv')
            save_loss_csv(save_file, pd.concat(scan_valid_loss), groupby=arg_names + ['ibatch'])
            logger.info(f'Saved valid curve: {save_file}')

            save_file = args.save_dir / (save_prefix + '_best.csv')
            scan_best_loss.to_csv(save_file, index=False, float_format='%.4g')
            logger.info(f'Saved scan summary: {save_file}')
        else:
            logger.info(f'scan results are not saved with args.save_dir: {args.save_dir}')

    return scan_best_loss


def fly(args, **kwargs):
    """  """
    if not isinstance(args, misc.Struct):
        args = misc.Struct(vars(args))

    # check local configuration json
    config_json = path.cwd() / 'config.json'
    if config_json.exists():
        logger.info(f'Loading local configuration json: {config_json}')
        args_local = gwio.json2dict(config_json)
        # print(json.dumps(args_local, indent=4))
        print(gwio.json_str(args_local))
        args.update(args_local)

    args.update(kwargs)
    logger.info(f'Applying kwargs:')
    print(gwio.json_str(kwargs))

    args = autoconfig_args(args) # resolve inconsistencies
    args.update(kwargs) # reapply...

    action_list = args.action
    if isinstance(action_list, str): action_list = [action_list]

    # resolve the load_dir use for loading purpose only
    if args.load_dir:
        if isinstance(args.load_dir, str): args.load_dir = path(args.load_dir)
        if not args.load_dir.exists():
            logger.critical(f'Model directory: {args.load_dir} does not exist, fail to load!')
            logger.critical(f'Use --save_dir {args.load_dir} if intended for saving...')
            sys.exit(1)
    elif 'predict' in sys.argv or 'validate' in sys.argv: # not really needed any more
        # beter load a model for validation and test/predict, set default
        args.load_dir = gwio.last_backup_path(args.net)

    if args.load_dir: # load args.json
        logger.info(f'Loading model args from directory: {args.load_dir}')
        if (args.load_dir / 'args.json').exists():
            args.update(gwio.json2dict(fname='args.json', fdir=args.load_dir))
            args.update(kwargs) # kwargs overwrite the model args!!!
            if isinstance(args.load_dir, str): args.load_dir = path(args.load_dir)
        else:
            logger.warning(f'args.json not found in {args.load_dir}, ' + \
                        'using default/command line args!!')

    if not args.save_dir: # set up save_dir
        args.save_dir = args.load_dir if args.load_dir else gwio.next_backup_path(args.net)
    if isinstance(args.save_dir, str): args.save_dir = path(args.save_dir)
    if not args.save_dir.exists(): args.save_dir.mkdir(parents=True)

    logger.info(f'Results will be saved in: \033[0;46m{args.save_dir}\033[0m')
    gwio.dict2json(vars(args), fname='last.json', fdir=path.cwd())

    if 'summary' in action_list or 'summarize' in action_list or 'view' in action_list:
        model = get_model(args)

    if 'train' in action_list:
        model = get_model(args)
        if args.resume and args.load_dir:
            state_dict_load(model, fdir=args.load_dir)

        args.net_src_file = save_net_pycode(args.net_src_file, args.save_dir)
        gwio.dict2json(vars(args), fname='args.json', fdir=args.save_dir)

        midata = get_midata(args)
        train_data, valid_data = train_test_split(midata, 
                    test_size=args.test_size, random_state=args.split_seed)

        callback_func = func_partial(validate_in_train, midata=valid_data,
            save_dir=args.save_dir, verbose=args.verbose)

        train(model, train_data, validate_callback=callback_func)

        # plot or do visualDL for paddle

    if 'cross_validate' in action_list:
        model = get_model(args)
        if args.resume and args.load_dir:
            state_dict_load(model, fdir=args.load_dir)

        args.net_src_file = save_net_pycode(args.net_src_file, args.save_dir)
        gwio.dict2json(vars(args), fname='args.json', fdir=args.save_dir)

        midata = get_midata(args)
        train_data, valid_data = train_test_split(midata, test_size=args.test_size, 
                    random_state=args.split_seed)

        # train as usual first
        callback_func = func_partial(validate_in_train, midata=valid_data,
            save_dir=args.save_dir, verbose=args.verbose)

        train(model, train_data, validate_callback=callback_func)

        # cross-validation training
        num_seqs = len(train_data)
        data_indices = np.linspace(0, num_seqs-1, num=num_seqs, dtype=int)
        data_indices = np.random.permutation(data_indices)

        num_xvalids = num_seqs // args.num_cvs
        xvalid_dirs = []
        for i in range(args.num_cvs):
            valid_data_xv = [train_data[data_indices[j]] for j in
                range(i*num_xvalids, (i+1)*num_xvalids)]
            train_data_xv = [train_data[data_indices[j]] for j in
                itertools.chain(range(0, i*num_xvalids), range((i+1)*num_xvalids, num_seqs))]

            save_dir_xv = args.save_dir / f'xvalid_{i}'
            xvalid_dirs.append(save_dir_xv)

            callback_func = func_partial(validate_in_train, midata=valid_data_xv,
                save_dir=save_dir_xv, verbose=args.verbose)
            model = get_model(args)

            train(model, train_data_xv, save_dir=save_dir_xv, validate_callback=callback_func)

    if 'scan_data' in action_list:
        # if args.resume and args.load_dir:
            # state_dict_load(model, fdir=args.load_dir)

        gwio.dict2json(vars(args), fname='args.json', fdir=args.save_dir)
        args.net_src_file = save_net_pycode(args.net_src_file, args.save_dir)

        midata = get_midata(args)
        train_data, valid_data = train_test_split(midata, test_size=args.test_size,
                    random_state=args.split_seed)

        args.validate_callback = func_partial(validate_in_train, midata=valid_data,
                save_dir=args.save_dir, verbose=args.verbose)

        scan_report = scan_data_args(args, train_data,
            valid_data = None,
            data_sizes = args.data_sizes, # [1,2,4], # 'auto',
            batch_sizes = args.batch_sizes, #[1,2,4], # 'auto',
        )

    if 'scout_args' in action_list:
        # if args.resume and args.load_dir:
            # state_dict_load(model, fdir=args.load_dir)

        if len(args.arg_values) != len(args.arg_names):
            logger.critical('arg names and values must of the same length!')

        args.net_src_file = save_net_pycode(args.net_src_file, args.save_dir)
        gwio.dict2json(vars(args), fname='args.json', fdir=args.save_dir)

        if args.rebake_midata:
            midata = load_pkldata(args)
            valid_data, train_data = random_split_dict(midata, size=args.test_size)
        else:
            midata = get_midata(args)
            train_data, valid_data = train_test_split(midata, test_size=args.test_size, 
                        random_state=args.split_seed)

        # will be overwritten in scout_args() if args.rebake_midata == True
        args.validate_callback = func_partial(validate_in_train, midata=valid_data,
                save_dir=args.save_dir, verbose=args.verbose)

        num_args = len(args.arg_names)
        # arg_values is a list of strings, e.g, ['1,2,3', '4,5,6']
        args.arg_values = [[float(_s) for _s in _v.split(',')] for _v in args.arg_values]
        if isinstance(args.arg_scales, int):
            args.arg_scales = [args.arg_scales] * num_args
        # pad with edge values if needed
        args.arg_scales = np.pad(args.arg_scales, (0, num_args - len(args.arg_scales)), mode='edge')

        passed_arg_values = args.arg_values
        if args.grid_search:
            args.arg_values = [np.sort(_v) for _v in args.arg_values]
        else: # generate a list of values
            args.arg_values = []
            for _i, min_max in enumerate(passed_arg_values):
                if args.arg_scales[_i] == 0:
                    args.arg_values.append(min_max[0] +
                        np.sort(np.random.random_sample(args.num_scouts)) * (min_max[-1] - min_max[0])
                    )
                else:
                    args.arg_values.append(np.exp(np.log(min_max[0]) +
                        np.sort(np.random.random_sample(args.num_scouts)) * np.log(min_max[-1] / min_max[0])
                    ))

        best_valid_loss = np.inf
        passed_arg_values = args.arg_values
        arg_values = args.arg_values
        master_save_dir = args.save_dir.resolve().as_posix()
        ispawn = 0
        while True:
            if ispawn == 0:
                save_dir = path(master_save_dir)
            else:
                save_dir = path(master_save_dir) / f'spawn_{ispawn}'

            scout_best_loss = scout_args(args, train_data, valid_data,
                arg_names = args.arg_names,
                arg_values = arg_values,
                grid_search = args.grid_search,
                save_dir = save_dir,
                #     arg_names = ['learning_rate', 'l2decay'],
                #     arg_values = [[1e-5, 1e-4, 1e-3, 1e-2], [1e-4, 1e-2]],
            )

            if not args.spawn_search: break #!!!! stop here uness spawn_search
            ispawn += 1

            # get arg_values giving the best valid_loss
            imin = scout_best_loss.valid_loss.argmin()
            if best_valid_loss < scout_best_loss.valid_loss[imin]:
                logger.info(f'Best spawned args found with valid_loss: {best_valid_loss}')
                break
            else:
                best_valid_loss = scout_best_loss.valid_loss[imin]

            # get the arg values giving the best loss
            arg_values_best = scout_best_loss.loc[imin, args.arg_names].to_numpy()
            # mutate from the best arg_values
            arg_values = []
            for _i in range(args.num_spawns):
                arg_values.append(arg_values_best.copy())

                iarg = np.random.randint(low=0, high=num_args)
                # make a random change, better stay within the confines of the input
                if args.arg_scales[iarg] == 0: # linear
                    arg_values[-1][iarg] = passed_arg_values[iarg][0] + \
                        np.random.rand() * (passed_arg_values[iarg][-1] - passed_arg_values[iarg][0])
                else:
                    arg_values[-1][iarg] = np.exp(np.log(passed_arg_values[iarg][0]) + \
                        np.random.rand() * np.log(passed_arg_values[iarg][-1] / passed_arg_values[iarg][0]))

            # get ready for the next scout run
            arg_values = list(zip(*arg_values))

    if 'average_model' in action_list:
        pkldata = load_pkldata(args)

        num_seqs = len(pkldata['seq'])
        num_models = len(args.model_dirs)

        valid_loss_models = np.ones((num_models,), dtype=np.float32)
        y_output_models = [] # np.array((num_models, num_data, midata[0][0].shape[0]), dtype=np.float32)
        as_label_models = []
        loss_vs_seq_models = np.zeros((num_models, num_seqs), dtype=np.float32)
        std_vs_seq_models = np.zeros((num_models, num_seqs), dtype=np.float32)
        
        logger.info(f'Averaging models in directories:{args.model_dirs}')
        for imodel, load_dir in enumerate(args.model_dirs):
            logger.info(f'Loading model from directory: {str(load_dir)} ...')
            
            if isinstance(load_dir, str): load_dir = path(load_dir)
            if args.best_earlystop: # use the best earlystop model
                load_dir = list(load_dir.glob('earlystop_*'))
                # directory name contains the validation loss
                loss_values = np.array([float(_dir.name.split('_')[-1]) for _dir in load_dir])
                idx_min = loss_values.argmin()
                load_dir = load_dir[idx_min]
                valid_loss_models[imodel] = loss_values[idx_min]
                logger.info(f'Found best earlystop: {load_dir} with valid loss: {valid_loss_models[imodel]}')

            # create and restore the model
            model_args, _ = parse_args2(['average_model'])
            model_args.update(gwio.json2dict(fname='args.json', fdir=load_dir))
            model_args.update(kwargs) # kwargs overwrite the model args!!!
            model_args.load_dir = path(load_dir)
            model = get_model(model_args)
            state_dict_load(model, fdir=load_dir)

            midata = bake_midata(pkldata, model_args)
            y_truth = [midata[i][-1] for i in range(num_seqs)]

            # rmsd_curve = validate(model, midata, shuffle=False, save_dir=None)

            # compute ymodel, a list of predicted numpy arrays for each
            y_output, seqs_len = predict(model, midata, shuffle=False, batch_size=args.batch_size,
                        save_dir=None)
            y_output_models.append(y_output)
            as_label_models.append(model.loss_fn[0].as_label(y_output).numpy())

            # compute loss
            if np.array_equal(as_label_models[-1][0].shape, y_truth[0].shape): 
                loss_vs_seq, std_vs_seq = compute_loss(model.loss_fn, y_output, y_truth,
                            batch_size=args.batch_size, seqs_len=seqs_len, shuffle=False,
                            loss_padding=args.loss_padding, loss_sqrt=args.loss_sqrt)

                logger.info(f'Model: {imodel} loss: \033[0;46m{loss_vs_seq.mean():6.4f}\033[0m' +
                            f' std: {std_vs_seq.mean():6.4f}')

                loss_vs_seq_models[imodel] = loss_vs_seq
                std_vs_seq_models[imodel] = std_vs_seq
            else:
                loss_vs_seq_models[imodel] = 1
                std_vs_seq_models[imodel] = 1

        # zip to get each item to be a list of model predictions for one seqence
        y_output_models = zip(*y_output_models)
        y_output_models = zip(*as_label_models)
        # stack to get one numpy array for each seq, then get the averaged model predictions!
        model_weights = 1.0 / valid_loss_models
        model_weights = model_weights / model_weights.sum()
        if num_seqs > 23:
            mpool = mpi.Pool(processes=int(mpi.cpu_count() * 0.8))
            y_output_models = mpool.map(np.stack, y_output_models)

            y_model_aver = mpool.map(func_partial(np.average, axis=0, weights=model_weights),
                                    y_output_models)
            mpool.close()
        else:
            y_output_models = list(map(np.stack, y_output_models))
            y_model_aver = list(map(func_partial(np.average, axis=0, weights=model_weights),
                                    y_output_models))

        if np.array_equal(y_model_aver[0].shape, y_truth[0].shape):
            loss_model_aver, std_model_aver = compute_loss( #model.loss_fn
                        SeqLossFn_P2P(F.mse_loss, name='mse', reduction='none'),
                        y_model_aver, y_truth, batch_size=args.batch_size, seqs_len=seqs_len,
                        shuffle=False, loss_padding=args.loss_padding, loss_sqrt=args.loss_sqrt)

            logger.info(f'Model losses: {loss_vs_seq_models.mean(axis=1)}')
            logger.info(f'Model averaged loss: \033[0;46m{loss_vs_seq_models.mean():6.4f}\033[0m' +
                        f' std: {std_vs_seq_models.mean():6.4f}')
            logger.info(f'Averaged model loss: \033[0;46m{loss_model_aver.mean():6.4f}\033[0m' +
                        f' std: {std_model_aver.mean():6.4f}')

        # save the average model results
        logger.info(f'Saving the averaged model estimates in: {str(args.save_dir)}')

        save_model_prediction(y_model_aver, args.save_dir, seqs_len=seqs_len, istart=1)

    if 'validate' in action_list:
        model = get_model(args)
        if args.load_dir:
            state_dict_load(model, fdir=args.load_dir)
        valid_data = get_midata(args)
        rmsd_curve = validate(model, valid_data, shuffle=False, save_dir=args.save_dir)

    if 'predict' in action_list:
        model = get_model(args)
        if args.load_dir:
            state_dict_load(model, fdir=args.load_dir)
        predict_data = get_midata(args)
        predict(model, predict_data, shuffle=False, save_dir=args.save_dir / 'predict.files')


if __name__ == '__main__' :
    """  """
    sys.setrecursionlimit(int(1e4))
    # if sys.stdin.isatty():     # if running from terminal
    args, argv_dict = parse_args2(sys.argv[1:])
    # else: # run from vscode # os.chdir()
        # args, argv_dict = parse_args(['-h'])

    misc.logging_config(logging, logfile=args.log, lineno=False, level=args.verbose)
    args.argv = " ".join(sys.argv[1:])
    logger.info(f'{path(__file__).name} {args.argv}')

    fly(args, **argv_dict)