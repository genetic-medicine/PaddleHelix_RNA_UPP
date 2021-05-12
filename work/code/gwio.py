#!/usr/bin/env python

import os
import re
import sys
import glob
import logging
import json
import functools
import pickle
import gzip
import bz2
# import lzma
# import brotli
from pathlib import Path as path
import numpy as np
import pandas as pd

# homebrew
import misc

logger = logging.getLogger(__name__)


def str2filename(str_in):
    return str_in.replace('\n', '-').replace(':[', '-').replace(',', '-').replace('];', '').replace('/', '%').strip()


def last_backup_path(fname):
    """ fname is returned if not exists """
    if isinstance(fname, str): fname = path(fname)

    if fname.exists():
        all_backups = list(fname.parent.glob(fname.name + '.[0-9]*'))
        if len(all_backups):
            suffixes = np.array([int(_f.suffix[1:]) for _f in all_backups])
            ilast = suffixes.argmax()

            fname = all_backups[ilast]

    return fname


def next_backup_path(fname):
    if isinstance(fname, str): fname = path(fname)

    if not fname.exists(): # new file
        return fname
    else:
        backup_file = last_backup_path(fname)

        if backup_file.samefile(fname): # no backup exists
            new_suffix = backup_file.suffix + '.1'
        else:
            new_suffix = '.' + str(int(backup_file.suffix[1:]) + 1)

        return backup_file.with_suffix(new_suffix)


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def pickle_squeeze(data, pkl_file, fmt='lzma'):
    """ looks like compress_pickle may be the way to go """
    if isinstance(pkl_file, str): pkl_file = path(pkl_file)

    if type(fmt) not in (list, tuple): fmt = [fmt]
    if len(fmt): fmt = [_s.lower() for _s in fmt]

    with pkl_file.open('wb') as hfile:
        logger.debug(f'Storing pickle: {pkl_file}')
        pickle.dump(data, hfile, -1)

    pkl_fname = pkl_file.as_posix()
    if 'gz' in fmt or 'gzip' in fmt:
        zip_file = pkl_fname + '.gz'
        logger.debug(f'Zipping and storing pickle: {zip_file}')
        with gzip.open(zip_file, 'wb') as hfile:
            pickle.dump(data, hfile, -1)

    if 'bz' in fmt or 'bz2' in fmt:
        zip_file = pkl_fname + '.pbz2'
        logger.debug(f'Zipping and storing pickle: {zip_file}')
        with bz2.BZ2File(zip_file, 'wb') as hfile:
            pickle.dump(data, hfile, -1)

    # if 'lzma' in fmt:
    #     zip_file = pkl_fname + '.xz'
    #     logger.debug(f'Zipping and storing pickle: {zip_file}')
    #     with lzma.open(zip_file, 'wb') as hfile:
    #         pickle.dump(data, hfile, -1)

    # with open('no_compression.pickle', 'rb') as f:
    #     pdata = f.read()
    #     with open('brotli_test.bt', 'wb') as b:
    #         b.write(brotli.compress(pdata))


def dict2json(dict_in, fname='args.json', fdir=path.cwd(), indent=(4, None)):
    """ if indent is a list, customized json encoder is used """
    if isinstance(fdir, str): fdir = path(fdir)
    fdir.mkdir(parents=True, exist_ok=True)
    
    json_file = fdir / fname

    if type(indent) in (list, tuple):
        with open(json_file, 'w') as hfile:
            hfile.writelines(json_str(dict_in, indent=indent[0]))
        return

    # serialize or remove keys from dict_in
    dict_bkp = {} # back up 
    keys_rmv = [] # only store names
    for key, val in dict_in.items():
        if not is_jsonable(val):
            dict_bkp[key] = val # save the original

            if isinstance(val, path):
                dict_in[key] = val.as_posix()
            else:
                keys_rmv.append(key)
    if len(keys_rmv):
        logger.info(f'The following keys are not jsonable: {keys_rmv}')
        for key in keys_rmv: dict_in.pop(key)
    
    logger.info(f'Saving json file: {json_file}...')

    with open(json_file, 'w') as hfile:
        if len(keys_rmv):
            pass
            # logger.warning(f'The following keys are excluded from JSON: {keys_rmv}')
            # hfile.write(f'# Keys not jsonable: {keys_rmv}\n')
        # if indent and oneline_list:
        #     out_list = json.dumps(dict_in, indent=indent)
        #     out_list = re.sub(r'([\t\f\v\r\n]+)([^:\[\]]+),[\s\$]+', r'\1\2, ', out_list)
        #     out_list = re.sub(r'([\t\f\v\r\n]+)([^:\[\]]+),[^[:print:]]+', r'\1\2, ', out_list)
        #     # out_list = re.sub(r'([\t\f\v\r\n]+)\s+([^:\[\]]+),\s+', r'\1\2, ', out_list)
        #     # out_list = re.sub(r'([^[:print:]]+)\s+([^:\[\]]+),\s+', r'\1\2, ', out_list)
        #     out_list = re.sub(r'": \[\s+', '": [', out_list)
        #     # out_list = re.sub(r'",\s+', '", ', out_list)
        #     out_list = re.sub(r'\s+\],', '],', out_list)
        #     hfile.writelines(out_list)
        # else:
        json.dump(dict_in, hfile, indent=indent)

    dict_in.update(dict_bkp) # restore backup


def json_str(val_in, indent=3, sep=[', ', ': '], space=' ', newline='\n', level=0):
    """ adopted from https://stackoverflow.com/questions/10097477/python-json-array-newlines """

    INDENT = indent
    SPACE = space
    NEWLINE = newline
    SEP = sep

    def to_json(o, level=0, ):
        ret = ""
        if isinstance(o, dict):
            ret += "{" + NEWLINE
            comma = ""
            for k,v in o.items():
                ret += comma
                comma = ",\n"
                ret += SPACE * INDENT * (level+1)
                ret += f'"{str(k)}"{SEP[-1]}'
                ret += to_json(v, level + 1)

            ret += NEWLINE + SPACE * INDENT * level + "}"
        elif isinstance(o, str):
            ret += '"' + o + '"'
        elif isinstance(o, list):
            ret += "[" + SEP[0].join([to_json(e, level+1) for e in o]) + "]"
        elif isinstance(o, bool):
            ret += "true" if o else "false"
        elif isinstance(o, int) or isinstance(o, np.integer):
            ret += str(o)
        elif isinstance(o, float) or isinstance(o, np.float):
            ret += '%.7g' % o
        elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.integer):
            ret += "[" + SEP[0].join(map(str, o.flatten().tolist())) + "]"
        elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.inexact):
            ret += "[" + SEP[0].join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
        elif isinstance(o, path):
            ret += f'"{o.resolve().as_posix()}"'
        elif isinstance(o, functools.partial):
            ret += f'"{o.func}"' # str(o) gives all parameters
        elif type(o) in ('module', 'function'):
            ret += f'"{str(o)}"'
        elif o is None:
            ret += 'null'
        else:
            # raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
            logger.warning(f'Unknow type {str(type(o))} of {o} for json serialization!')
            ret += 'null'
        return ret

    return to_json(val_in, level=level)


def json2dict(fname='args.json', fdir=path.cwd()):
    """  """
    if isinstance(fdir, str): fdir = path(fdir)
    json_file = fdir / fname

    logger.info(f'Loading json file: {json_file}...')
    with open(json_file, 'r') as hfile:
        dict_out = json.load(hfile)

    return dict_out


def get_file_lines(fname, fdir='', strip=False, keep_empty=True,
                   comment='#', keep_comment=True):
    """  """
    flines = []
    if type(fname) not in (list, tuple): fname = [fname]
    for onefile in fname:
        with open(os.path.join(fdir, onefile), 'r') as hfile:
            if strip:
                flines.extend([_s.strip() for _s in hfile.readlines()])
            else:
                flines.extend(hfile.readlines())

    if keep_comment and keep_empty:
        return flines
    elif keep_comment:
        return [_s for _s in flines if len(_s) > 0]
    elif keep_empty:
        return [_s for _s in flines if not _s.startswith(comment)]
    else:
        return [_s for _s in flines if len(_s) > 0 and _s[0] != comment]


def read_dataframe(csvfiles, fmt='infer', sep=',', header=0, skiprows=0,
                joinaxis=0, sort=False, return_file=False):
    """  """
    # csvfiles = misc.unpack_list_tuple(csvfiles)
    if type(csvfiles) not in (list, tuple): csvfiles =[csvfiles]
    filenames = [filename for _f in csvfiles for filename in glob.glob(str(_f))]

    num_files = len(filenames)
    csvdata = []
    for ifile, filename in enumerate(filenames):
        logger.info(f"Reading csv file: <{filename}> ({ifile + 1} out of {num_files})")
        # infer sep
        if fmt == 'infer':
            suffix = filename.split('.')[-1].lower()
            if suffix == 'csv':
                sep = ','
            elif suffix in ['txt', 'upp', 'dat']:
                header = None
                sep = r'\s+'

        csvdata.append(pd.read_csv(filename, sep=sep, header=header, skiprows=skiprows))
            
    if len(csvdata):
        csvdata =  pd.concat(csvdata, axis=joinaxis, sort=sort)
        logger.info(f'The combined data have dimensions: {csvdata.shape}')
    else:
        logger.info(f'Files {csvfiles} not found or unreadable!!!')

    if any(csvdata.columns.duplicated()):
        logger.info('Contains duplicated column names, reset to integers...')
        csvdata.columns = list(range(csvdata.shape[1]))

    if return_file:
        return csvdata, filenames
    else:
        return csvdata


def setcwd():
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    print('Current working directory: ' + os.getcwd())
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    print('Changing to the file directory: ' + dname)
    os.chdir(dname)


def showinfo(infostr="checking showinfo", infotype='info'):
    """Display information during execution of a function. The name of
    the function will be shown!

    infostr  -- the information string
    infotype -- the kind of information: info, warning, error

    Return None
    """
    from traceback import extract_stack
    callerinfo = extract_stack()[-2]
    # print extract_stack()
    if callerinfo[2] == '<module>':
        callername = os.path.basename(callerinfo[0])
        if callername[0] == '<':
            callername = 'gwio'
    else:
        callername = callerinfo[2]
    print("[%s::%s] %s" % (infotype.upper(), callername, infostr))
    return
