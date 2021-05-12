#!/usr/bin/env python
# %%
import re
import numpy as np
import multiprocessing as mpi
import time
from datetime import datetime
import logging
import colorlog
from collections import defaultdict
logger = logging.getLogger(__name__)


class MainError(Exception):
    pass


class Struct():
    def __init__(self, in_dict=dict(), **kwargs):
        self.__dict__.update(in_dict)
        self.__dict__.update(kwargs)

    def update(self, in_dict=dict(), **kwargs):
        self.__dict__.update(in_dict)
        self.__dict__.update(kwargs)


def mpi_map(hfunc, data_in, num_cpus=0.8, starmap=False):
    """  """
    if 0 < num_cpus < 1:
        num_cpus = round(mpi.cpu_count() * 0.8)
    elif num_cpus > mpi.cpu_count():
        num_cpus = mpi.cpu_count() - 1

    logger.info(f'Using multiprocessing, cpu count: {num_cpus}')
    mpool = mpi.Pool(processes=num_cpus)
    if starmap:
        data_out = mpool.starmap(hfunc, data_in)
    else:
        data_out = mpool.map(hfunc, data_in)
    mpool.close()

    return data_out


def str_deblank(str_in):
    """ see https://stackoverflow.com/questions/3739909/how-to-strip-all-whitespace-from-string """
    return re.sub(r'(\s|\u180B|\u200B|\u200C|\u200D|\u2060|\uFEFF)+', '', str_in)


def str_color(str_in, color='none', bkg='none', style='normal'):
    """ escape char: \033 \e \x1B  """
    style_dict = defaultdict(lambda: 0)
    style_dict.update(dict(
        normal = 0,
        bold = 1,
        dim = 2,
        underline = 4,
        blink = 5,
        reverse = 7,
        hidden = 8,
    ))

    fg_color_dict = defaultdict(lambda: 39)
    fg_color_dict.update(dict(
        black = 30,
        red = 31,
        green = 32,
        yellow = 33,
        blue = 34,
        magenta = 35,
        cyan = 36,
        light_gray = 37,
        dark_gray = 90,
        light_red = 91,
        light_green = 92,
        light_yellow = 93,
        light_blue = 94,
        light_magenta = 95,
        light_cyan = 96,
        white = 97))
    bg_color_dict = defaultdict(lambda: 49)
    for k, v in fg_color_dict.items():
        bg_color_dict[k] = v + 10

    # \033[0;46m{args.save_dir}\033[0m'
    str_out = f'\033[{style_dict[style]};{fg_color_dict[color]};{bg_color_dict[bkg]}m{str_in}\033[0m'

    return str_out


def logging_config(logging, logfile=None, lineno=True, funcname=True, level=1):
    """  """

    handler = logging.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        # "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        # datefmt=None,
        "%(log_color)s%(levelname)s %(cyan)s%(asctime)s" + \
               ("%(yellow)s @%(funcName)s" if funcname else "") + \
               (" [%(lineno)4d]" if lineno else "") + \
               ":: %(reset)s%(message)s",
        datefmt='%m/%d %H:%M:%S',  # %Y-%m-%d
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%',
    )
    handler.setFormatter(formatter)

    logging.basicConfig(
        format="%(levelname)s %(asctime)s" + \
               (" @%(funcName)s" if funcname else "") + \
               (" [%(lineno)4d]" if lineno else "") + \
               ":: %(message)s",
        datefmt='%m/%d %H:%M:%S',  # %Y-%m-%d
        handlers = [logging.FileHandler(logfile), handler]
            if logfile else [handler],
        )

    logging.getLogger().setLevel([logging.WARNING, logging.INFO, logging.DEBUG][level])


def logger_setlevel(logger, level=1):
    """  """
    logger.setLevel([logging.WARNING, logging.INFO, logging.DEBUG][level])


def argv_optargs(argv, args=Struct()):
    """ generate a struct with all optional arguments in argv.
        If a key exists in args, its value is copied.
    Caution: it only look for --OPTION """
    keys = [_key[2:] for _key in argv if _key.startswith('--')]
    cline_args = dict.fromkeys(keys)
    for _key in set(keys).intersection(vars(args).keys()) :
        cline_args[_key] = vars(args)[_key]
    return Struct(**cline_args)


def time_elapsed():
    pass


def time_now_str(fmt="%I:%M%p %B %d, %Y"):
    return datetime.now().strftime(fmt)


def round_sig(x, sig=2):
    return round(x, sig-int(np.floor(np.log10(abs(x))))-1)


def alphanum_start(str_in):
    i = re.search(r'\W+', str_in).start()
    return str_in[0: i]


def get_list_index(vlist, v, offset=0, last=False):
    # get_index = lambda vlist, v : [vlist.index(v)] if v in vlist else []
    try:
        if last:
            v_reversed = vlist[-1::-1]  # vlist.reverse() changes vlist
            return [len(v_reversed) - 1 - v_reversed.index(v) + offset]
        else:
            return [vlist.index(v) + offset]
    except ValueError:
        return []


def unpack_list_tuple(*list_in):
    if len(list_in) == 0:
        return []
    list_out = []
    types2unpack = (list, tuple)
    for _item in list_in:
        if type(_item) in types2unpack:
            list_out.extend(unpack_list_tuple(*_item))
        else:
            list_out.append(_item)
    return list_out


def fuzzy_name(str_in):
    """ convert to lower cases and remove common delimiters """
    if isinstance(str_in, str):
        return str_in.lower().replace('_', '').replace('-', '')
    else:
        return [_s.lower().replace('_', '').replace('-', '')
                if isinstance(_s, str) else _s for _s in str_in]


def zflatten2xyz(z, x=None, y=None):
    """ flatten an nxm 2D array to [x, y, z] of shape=(n*m, 3)"""
    if x is None:
        x = np.arrange(0, z.shape[0], step=1)
    if y is None:
        y = np.arrange(0, z.shape[1], step=1)
    xlen = len(x)
    ylen = len(y)
    assert z.shape[0] == xlen and z.shape[1] == ylen, 'check dimensions!!!'

    xx, yy = np.meshgrid(x, y)
    xx = xx.T
    yy = yy.T  # meshgrid take the second dimension as x

    xylen = xlen*ylen
    return np.concatenate((xx.reshape((xylen, 1)),
                           yy.reshape((xylen, 1)),
                           z.reshape((xylen, 1))), axis=1)


def zflatten2xyz_debug():
    x = np.arange(0, 5, step=1)
    y = np.arange(0, 4, step=1)

    xx, yy = np.meshgrid(x, y)

    z = xx.T*10 + yy.T
    print('z[2,3]:]', z[2, 3])
    print('x', x)
    print('y', y)
    print('z', z)
    xyz = zflatten2xyz(z, x=x, y=y)
    print(xyz)


if __name__ == '__main__':
    zflatten2xyz_debug()
