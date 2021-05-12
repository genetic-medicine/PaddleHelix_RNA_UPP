# %%
import functools
import os.path
import logging
from io import StringIO
import numpy as np
import pandas as pd
import multiprocessing as mpi
from collections import defaultdict, namedtuple
from pathlib import Path as path

# homebrew
import misc
from gwio import get_file_lines
from misc import get_list_index

logger = logging.getLogger(__name__)

# ======= convert structure into scalars/quants  =======
# defaultdict ADD a field if not present, undesired!
idx2res_dict = dict() # defaultdict(lambda: '-')
idx2res_dict.update({
    0: '-',
    1: 'A',
    2: 'U',
    3: 'C',
    4: 'G',
})
res2idx_dict = dict() # defaultdict(lambda: 0)
res2idx_dict.update(dict([(_v, _k) for _k, _v in idx2res_dict.items()]))

idx2dbn_dict = dict() # defaultdict(lambda: '-')
idx2dbn_dict.update({
    0: '-',
    1: '.',
    2: '(',
    3: ')',
})
dbn2idx_dict = dict() # defaultdict(lambda: 0)
dbn2idx_dict.update(dict([(_v, _k) for _k, _v in idx2dbn_dict.items()]))


def quant_rna_seq(seq, dbn=None, use_nn=0, use_dbn=False, length=None):
    """ use a single number for each seq/dbn
    0 denotes padding/start/end
    the largest number is currently not used
    """
    seq_len = len(seq)
    if not length:
        length = seq_len
    if not use_nn:
        use_nn = 0
    else:
        use_nn = int(use_nn)

    # deal with nearest eighbor inclusions
    seq = ('-' * use_nn) + seq + ('-' * use_nn)
    if use_dbn and dbn is not None:
        dbn = ('-' * use_nn) + dbn + ('-' * use_nn)
        assert len(dbn) == len(seq), 'seq and dbn must have the same length!'

    # first get the list of idx for each residue
    seq_emb = []
    nn_indices = list(range(-use_nn, use_nn + 1))

    # this is the shape/len for each dim
    dim_sizes = [len(res2idx_dict)] * (2 * use_nn + 1)
    if use_dbn and dbn is not None:
        dim_sizes.extend([len(dbn2idx_dict)] * (2 * use_nn + 1))

    for i in range(use_nn, min([length, seq_len]) + use_nn):
        # first each residue is represented by a list of indices for seq/dbn/etc
        res_emb = [res2idx_dict.get(seq[i + _i], 0) for _i in nn_indices]

        if use_dbn and dbn is not None:
            res_emb.extend([dbn2idx_dict.get(dbn[i + _i], 0) for _i in nn_indices])

        # if use_attr: (not necessary for using attributes)
            # res_embeded.append(res2onehot_attr_dict[seq[i]])
        seq_emb.append(np.array(res_emb, dtype=np.int32))

    seq_emb = np.stack(seq_emb, axis=0)
    dim_sizes = np.array(dim_sizes, dtype=np.int32)
    # the mutiplier for each dim, 1 for the last dim
    dim_multplier = np.concatenate((np.flip(np.cumprod(np.flip(dim_sizes[1:]))),
                                      np.ones((1), dtype=np.int32)), axis=0)

    # quantize the vector for each residue
    seq_emb = np.matmul(seq_emb, dim_multplier)

    if len(seq_emb) < length:
        seq_emb = np.concatenate((seq_emb, np.zeros((length - len(seq_emb),), dtype=np.int32)))

    return seq_emb


def quant_rna_seq_old(seq, dbn=None, use_nn=0, use_dbn=False, length=None):
    """ use a single number for each seq/dbn """
    if not length:
        length = len(seq)

    if use_dbn and dbn is not None:
        num_dbn_idx = len(dbn2idx_dict)
        seq_quant = [(res2idx_dict[_s]-1) * num_dbn_idx +
                      dbn2idx_dict[dbn[_i]] if res2idx_dict[_s] else 0
                      for _i, _s in enumerate(seq[:length])]
    else:
        seq_quant = [res2idx_dict[_s] for _s in seq[:length]]

    if len(seq) < length:
        seq_quant.extend([0] * (length - len(seq)))

    return np.array(seq_quant, dtype=np.int32)

# ======= convert structure into vectors/embeddings =======
# The following dict was modified from e2efold code
res2onehot_dict = defaultdict(lambda: np.array([0, 0, 0, 0]))
res2onehot_dict.update({  # A? U? C? G?
    '-': np.array([0, 0, 0, 0], dtype=np.int32),
    '.': np.array([0, 0, 0, 0], dtype=np.int32),
    'A': np.array([1, 0, 0, 0], dtype=np.int32),
    'U': np.array([0, 1, 0, 0], dtype=np.int32),
    'C': np.array([0, 0, 1, 0], dtype=np.int32),
    'G': np.array([0, 0, 0, 1], dtype=np.int32),
    'N': np.array([1, 1, 1, 1], dtype=np.int32),
    'X': np.array([0, 0, 0, 0], dtype=np.int32),
    'I': np.array([0, 0, 0, 0], dtype=np.int32),
    'P': np.array([0, 0, 0, 0], dtype=np.int32),  # Phosphate only
    'M': np.array([1, 0, 1, 0], dtype=np.int32),  # aMide
    'K': np.array([0, 1, 0, 1], dtype=np.int32),  # Keto
    'R': np.array([1, 0, 0, 1], dtype=np.int32),  # puRine
    'Y': np.array([0, 1, 1, 0], dtype=np.int32),  # pYrimidine
    'W': np.array([1, 1, 0, 0], dtype=np.int32),  # Weak
    'S': np.array([0, 0, 1, 1], dtype=np.int32),  # Strong
    'V': np.array([1, 0, 1, 1], dtype=np.int32),  # not U/T
    'D': np.array([1, 1, 0, 1], dtype=np.int32),  # not C
    'B': np.array([0, 1, 1, 1], dtype=np.int32),  # not A
    'H': np.array([1, 1, 1, 0], dtype=np.int32),  # not G
})

# Additional features for five bases: A, U, C, G, -
res2onehot_attr_dict = defaultdict(lambda: np.array([0] * 8))
res2onehot_attr_dict.update({
    #    AU GC  GU M  K  S  R  Y more to add?
    '-': np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
    'A': np.array([1, 0, 0, 1, 0, 0, 1, 0], dtype=np.int32),
    'U': np.array([1, 0, 1, 0, 1, 1, 0, 1], dtype=np.int32),
    'C': np.array([0, 1, 0, 1, 0, 0, 0, 1], dtype=np.int32),
    'G': np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=np.int32),
})
# seq_trait_mat = np.array([
#     # AU? GC? GU? M? K? S?   more to add?
#     [1,   0,  0, 1, 0, 0],
#     [1,   0,  1, 0, 1, 1],
#     [0,   1,  0, 1, 0, 0],
#     [0,   1,  1, 0, 1, 1],
#     [0,   0,  0, 0, 0, 0],
# ], dtype=np.int32)

dbn2onehot_dict = defaultdict(lambda: np.array([0, 0, 0, 0]))
dbn2onehot_dict.update({
    '-': np.array([0, 0, 0, 0], dtype=np.int32),  # gap or padding
    '.': np.array([1, 0, 0, 0], dtype=np.int32),
    '(': np.array([0, 1, 0, 0], dtype=np.int32),
    ')': np.array([0, 0, 1, 0], dtype=np.int32),
})

def vector_rna_seq(seq, dbn=None, bpp=None, length=None,
                use_nn=0, use_attr=False, use_dbn=False, use_bpp=False,
                res2vec_dict=res2onehot_dict,
                dbn2vec_dict=dbn2onehot_dict,
                attr2vec_dict=res2onehot_attr_dict):
    """ embed each seq/dbn with a vector """
    seq_len = len(seq)
    if not length:
        length = seq_len
    if not use_nn:
        use_nn = 0
    else:
        use_nn = int(use_nn)

    # pad for nearest neighbor inclusion
    seq = ('-' * use_nn) + seq + ('-' * use_nn)
    if use_dbn is not None and dbn is not None:
        dbn = ('-' * use_nn) + dbn + ('-' * use_nn)
        assert len(dbn) == len(seq), 'seq and dbn must have the same length!'

    seq_emb = []
    nn_indices = list(range(-use_nn, use_nn + 1))

    for i in range(use_nn, min([length, seq_len]) + use_nn):
        res_emb = [res2vec_dict[seq[i + _i]] for _i in nn_indices]

        if use_dbn and dbn is not None:
            res_emb.extend([dbn2vec_dict[dbn[i + _i]] for _i in nn_indices])

        if use_attr: # no nn for trait yet
            res_emb.append(attr2vec_dict[seq[i]])

        if use_bpp: # the base pair probability from linear_partition
            res_emb.append(bpp[i - use_nn])

        if isinstance(res_emb[0], np.ndarray):
            seq_emb.append(np.concatenate(res_emb))
        else:
            seq_emb.append(np.array(res_emb, dtype=np.int32))

    # for i in range(len(seq_emb), length):
        # seq_emb.append(np.zeros((seq_emb[0].shape[0]), dtype=np.int16))
    if length > len(seq_emb):
        seq_emb.extend(np.zeros((length - len(seq_emb), seq_emb[0].shape[0]), dtype=np.int32))

    seq_emb = np.stack(seq_emb, axis=0)

    # if use_attr and use_dbn and dbn is not None:
    #     seq_embeded = [np.concatenate((
    #         res2onehot_dict[_s],
    #         res2onehot_attr_dict[_s],
    #         dbn2onehot_dict[dbn[_i]],
    #     )) for _i, _s in enumerate(seq)]
    # elif use_attr and not use_dbn:
    #     seq_embeded = [np.concatenate((
    #         res2onehot_dict[_s],
    #         res2onehot_attr_dict[_s],
    #     )) for _s in seq]
    # elif use_dbn and dbn is not None:
    #     seq_embeded = [np.concatenate((
    #         res2onehot_dict[_s],
    #         dbn2onehot_dict[dbn[_i]],
    #     )) for _i, _s in enumerate(seq)]
    # else:
    #     seq_embeded = [res2onehot_dict[_s] for _s in seq]

    # if use_dbn and dbn is not None:
    #     for _i, _s in enumerate(dbn):
    #         seq_embeded[_i].extend(dbn2onehot_dict[_s])
    # rna_embeded = [np.concatenate((
    #                     res2onehot_dict[_s],
    #                     res2onehot_attr_dict[_s] if use_attr else [],
    #                     dbn2onehot_dict[dbn[_i]] if use_dbn else [],
    #                     )) for _i, _s in enumerate(seq[:length])]
    return seq_emb


def DEBUG_encoding():
    seq_debug = 'AUCG'*3 + '-'*3
    # seq_debug = np.random.randint(0, 5, size=50)
    dbn_debug = '.()'*4 + '-'*3
    print('Quant and Embed encoding examples:\n')
    print(f'SEQ: {seq_debug}')
    print(f'DBN: {dbn_debug}')
    print('\nQuant/scalar encoding')
    print(quant_rna_seq(seq_debug))
    print('\nQuant/scalar encoding fixed length=23')
    print(quant_rna_seq(seq_debug, length=23))
    print('\nQuant/scalar encoding with dbn')
    print(quant_rna_seq(seq_debug, dbn_debug, use_dbn=True, length=23))

    print('\nEmbending sequence only')
    print(vector_rna_seq(seq_debug, use_attr=False))
    print('\nEmbending sequence with dbn')
    print(vector_rna_seq(seq_debug, dbn=dbn_debug, use_dbn=True, use_attr=False))
    print('\nEmbending sequence with dbn and trait')
    print(vector_rna_seq(seq_debug, dbn=dbn_debug, use_dbn=True, length=23, use_attr=True))


def pairing_energy(na_pair):
    """ adopted from e2efold code """
    set_pair = set(na_pair)
    if set_pair == {'A', 'U'}:
        E = 2
    elif set_pair == {'G', 'C'}:
        E = 3
    elif set_pair == {'G', 'U'}:
        E = 0.8
    else:
        E = 0
    return E


def parse_bpseq_lines(str_lines, return_matrix=True):
    """ each line of bpseq contains 3 columns:
            resnum resname bp_resnum (or 0)
    returns
    """
    num_resids = len(str_lines)
    seq = ''
    bp = []
    bp_mat = np.zeros((num_resids, num_resids), dtype=np.int32)

    for i in range(num_resids):
        tokens = str_lines[i].split()
        if int(tokens[0]) != i + 1:
            logger.warning(f'residue number is corrupted: {str_lines[i]}')
            continue

        seq += tokens[1]

        j = int(tokens[2])
        if j == 0: continue

        bp.append([i + 1, j])
        bp_mat[[i, j - 1], [j - 1, i]] = 1

    return np.stack(bp), bp_mat, seq


def parse_upp_lines(upp_lines, l=None):
    """ Format: resnum upp
    Only parse up to the length l, if passed """

    num = l if l else len(upp_lines)
    if num < len(upp_lines):
        logger.warning(f'Passed length: {l} < # of upp lines: {len(upp_lines)}')

    upp_idx = np.zeros((num,), dtype=np.int32)
    upp_val = np.zeros((num,), dtype=np.float32)

    for i, s in enumerate(upp_lines[:num]):
        tokens = s.split()
        upp_idx[i] = int(tokens[0])
        upp_val[i] = float(tokens[1])

    # check continuity of upper_idx
    if upp_idx[0] != 1 or any(upp_idx[1:] - upp_idx[:-1] != 1):
        logger.critical("Indices are not continuous in UPP data!")

    return upp_val


def parse_sequence_lines(fasta_lines, fmt='fasta', id=True, seq=True, dbn=None, upp=None,
            return_namedtuple=False, return_dict=False):
    """ Only one sequence is contained in the lines.
    Requires:
            1) stripped at the ends
            2) no additional comment lines
    """
    fmt = fmt.lower()
    i = 0
    if id:
        if fmt == 'seq':
            i += 1 # ID is at one line after the last ";"
            id = fasta_lines[i]
        else:
            id = fasta_lines[i][1:]
        # id = id.replace(',', ':')
        i += 1
    else:
        id = ''

    # assume both the sequence and dbn are one-liners
    if seq:
        if not dbn and not upp: # use all remaining lines
            seq = ''.join(fasta_lines[i:])
        else:
            seq = fasta_lines[i]

        if fmt == 'seq': seq = seq[:-1] # remove 1 at the end

        seq = misc.str_deblank(seq)
        i += 1
    else:
        seq = ''

    if dbn:
        dbn = misc.str_deblank(fasta_lines[i])
        i += 1
    else:
        dbn = ''

    if upp:
        upp = parse_upp_lines(fasta_lines[i:])
    else:
        upp = np.empty(0, dtype=np.float32)

    if return_namedtuple:
        FastaInfo = namedtuple('FastaInfo', ['id', 'seq', 'dbn', 'upp'],
                defaults=[None]*4)
        return FastaInfo(id=id, seq=seq, dbn=dbn, upp=upp)
    elif return_dict:
        return dict(id=id, seq=seq, dbn=dbn, upp=upp)
    else:
        return (id, seq, dbn, upp)


def parse_ct_lines(ct_lines, is_file=False, return_dict=True):
    """  """
    if is_file:
        fname = ct_lines
        ct_lines = get_file_lines(ct_lines, keep_empty=False)
    else:
        fname = ''

    header_tokens = ct_lines[0].split()
    seq_len = int(header_tokens[0])
    id = ' '.join(header_tokens[1:])
    # id = id.replace(',', ':')

    df = pd.read_csv(StringIO('\n'.join(ct_lines[1:])), sep=r'\s+', skiprows=0,
                header=None, engine='c',
                names=['resnum', 'resname', 'nn_before', 'nn_after', 'resnum_bp', 'resnum2'])

    if seq_len == df.shape[0]:
        seq = df.resname.sum()
    else:
        seq = np.array(['N'] * seq_len)
        seq[df.resnum.to_numpy(dtype=np.int32) - 1] = df.resname.to_list()
        seq = ''.join(seq)
        logger.warning(f'Mismatching sequence length: {seq_len} and ct table: {df.shape[0]}' + \
                      f' for id: {id}, file: {fname}')

    # generate the list of contacts
    ct = df[['resnum', 'resnum_bp']].to_numpy(dtype=np.int32)

    # remove lines with resnum_bp = 0
    ct = ct[ct[:, 1] > 0]

    # concatenate just in case only half of the ct table is provided
    # ct = np.concatenate((ct, np.flip(ct, axis=1)), axis=0)

    # sort and removes redundant [j, i] pairs
    # ct = ct[ct[:, 1] > ct[:, 0]]
    ct = np.sort(ct, axis=1)
    ct = np.unique(ct, axis=0)

    # collect data into a dict
    seq_dict = dict(ct=ct, len=seq_len, seq=seq, id=id)

    if return_dict:
        return seq_dict
    else:
        return ct, id, seq, seq_len


def dbn2ct(dbn):
    stack = []
    ct = []

    for i, s in enumerate(dbn):
        if s == '(':
            stack.append(i + 1)
        elif s == ')':
            ct.append([stack.pop(), i + 1])

    if len(stack):
        logger.warning(f'umatched ()s in: {dbn}')

    return np.stack(ct)


def bpp2upp(bpp, l=None):
    """ bpp is the base-pair probabilities from linear_partition, [i, j, p]
        upp is the unpaired probabilities 
    """
    if len(bpp) == 0:
        return np.zeros(l, dtype=np.float32)
    if l is None: l = int(bpp.max())

    upp = np.zeros(l, dtype=np.float32)

    for i in [0, 1]:
        idx_sorted = np.argsort(bpp[:, i])
        bpp_sorted = bpp[idx_sorted]
        resnums_unique, idx_unique, resnum_counts = \
            np.unique(bpp_sorted[:, i].astype(int) - 1, return_index=True, return_counts=True)

        # add the first occurrence of the resum
        upp[resnums_unique] += bpp_sorted[idx_unique, 2]

        idx_unique = np.append(idx_unique, [bpp_sorted.shape[0]], axis=0)
        # additional base pairing for some bases
        for j in np.where(resnum_counts > 1)[0]:
            upp[resnums_unique[j]] += bpp_sorted[(idx_unique[j] + 1):idx_unique[j+1], 2].sum()

    return 1.0 - upp


def ct2dbn(ct, l=None):
    """ ct should be nx2 numpy array """
    if l is None: l = ct.max()

    dbn = np.array(['.'] * l)

    dbn[ct[:, 0] - 1] = '('
    dbn[ct[:, 1] - 1] = ')'

    return ''.join(dbn)


def ct2mat(ct, l=None):
    """ ct should be nx2 numpy array """
    if l is None: l = ct.max()

    ct_mat = np.zeros((l, l), dtype=np.int32)

    ict = ct -1

    ct_mat[ict[:, 0], ict[:, 1]] = 1
    ct_mat[ict[:, 1], ict[:, 0]] = 1

    return ct_mat


def mat2ct(ct_mat):
    """ ct should be nx2 numpy array """

    ct = np.stack(np.where(ct_mat == 1), axis=1) + 1
    ct = ct[ct[:, 1] > ct[:, 0]]

    return ct


def count_pseudoknot(bp, sorted=True):
    """ bp is nx2 numpy array
    condition: i1 < i2 < j1 < j2
    """
    if not sorted:
        bp = np.sort(bp, axis=1) # i < j for each bp
        bp = bp[bp[:, 0].argsort()] # i1 < i2 for all bps

    resnums = []
    # for a sorted pair list, only need to check i2 < j1 < j2
    for _i, j1 in enumerate(bp[:-1, 1]):
        _i += 1
        if np.logical_and(bp[_i:, 0] < j1, j1 < bp[_i:, 1]).any():
            resnums.append(j1)

    # get the number of continuous segaments
    if len(resnums):
        resnums = np.sort(np.array(resnums))
        return (np.diff(resnums) > 1).sum() + 1
    else:
        return 0


class SeqsData():
    def __init__(self, fnames=[], fmt='fasta', fdir='', **kwargs):
        self.file = []
        self.id = []   # a list of string
        self.type = [] # a list of string
        # 1D info
        self.seq = []  # a list of string
        self.upp = []  # a list of np.array(1d) (up paird probability)
        self.len = np.empty(0, dtype=np.int32) # a 1D np array
        self.nbreaks = np.empty(0, dtype=np.int32)  # a 1D np array
        self.npknots = np.empty(0, dtype=np.int32)  # a 1D np array

        # 2D info
        self.dbn = []  # a list of string (dot bracket notation)
        self.ct = []   # a list of np.array(2d) (contact table)
        self.ct_mat = [] # a list of 2D numpy array
        self.bp = []   # bpseq?
        self.bp_mat = []
        if isinstance(fnames, path) or len(fnames):
            if type(fnames) not in (list, tuple): fnames = [fnames]
            fdir = path(fdir)
            fnames = [fdir / _f for _f in fnames]

            if fmt.lower() in ['fasta', 'seq']:
                self.parse_sequence_file(fnames, fmt=fmt, **kwargs)
            elif fmt.lower() == 'ct':
                self.parse_ct_file(fnames, **kwargs)
            elif fmt.lower() == 'bpseq':
                self.parse_bpseq_file(fnames, **kwargs)
            else:
                logger.warning(f'file format: {fmt} not recognized!')

    def add_dummy_seq(self, num=1):
        """  """
        self.file.extend([''] * num)
        self.id.extend([''] * num)
        self.type.extend([''] * num)
        self.seq.extend([''] * num)
        self.upp.extend([np.empty(0, dtype=np.float32) for i in range(num)])
        self.len = np.pad(self.len, (0, num), 'constant', constant_values=(0, 0))
        self.nbreaks = np.pad(self.nbreaks, (0, num), 'constant', constant_values=(0, 0))
        self.npknots = np.pad(self.npknots, (0, num), 'constant', constant_values=(0, 0))

        self.dbn.extend([''] * num)
        self.ct.extend([np.empty(0, dtype=np.int) for _i in range(num)])
        self.bp.extend([np.empty(0, dtype=np.int) for _i in range(num)])
        self.ct_mat.extend([np.empty(0, dtype=np.int) for _i in range(num)])
        self.bp_mat.extend([np.empty(0, dtype=np.int) for _i in range(num)])

    def add_seq_from_dict(self, seqs_dict, istart=None, fname=None):
        """ seqs_dict is a dict (or list of dict).
            each dict contains one sequence only.
        """
        if type(seqs_dict) not in (list, tuple):
            seqs_dict = [seqs_dict]
        if istart is None:
            istart = len(self.seq) # index of self.id/seq/...
        if type(fname) not in (list, tuple):
            fname = [fname]

        num_seqs = len(seqs_dict)
        if (istart + num_seqs) > len(self.seq):
            self.add_dummy_seq(istart + num_seqs - len(self.seq))
        fname += [fname[-1]] * (num_seqs - len(fname))

        for i in range(num_seqs):
            iself = istart + i

            self.file[iself] = seqs_dict[i].get('fname', self.file[iself]) \
                                if fname[i] is None else fname[i]
            self.id[iself] =  seqs_dict[i].get('id', self.id[iself])
            self.seq[iself] = seqs_dict[i].get('seq', self.seq[iself])
            self.len[iself] = len(self.seq[iself])
            self.dbn[iself] = seqs_dict[i].get('dbn', self.dbn[iself])
            self.upp[iself] = seqs_dict[i].get('upp', self.upp[iself])
            self.len[iself] = seqs_dict[i].get('len', self.len[iself])

            self.ct[iself] = seqs_dict[i].get('ct', self.ct[iself])

    def parse_sequence_file(self, fnames, fmt='fasta', id=True, seq=True, dbn=None, upp=None,
                istart=None, **kwargs):
        """ the default is fasta file with id and seq fields """

        # set up parameters
        if isinstance(fnames, str): fnames = [fnames]
        fmt = fmt.lower()
        if fmt == 'fasta':
            id_tag = '>'
        elif fmt == 'seq':
            id_tag = ';'
        else:
            id_tag = None
            logger.critical(f'Unknow sequence file format: {fmt}')

        # read the files
        str_lines = get_file_lines(fnames, strip=True, keep_empty=False)
        num_lines = len(str_lines)
        logger.debug(f'File <{str(fnames)}> has {num_lines} of lines')

        # find all lines with id tags ------ [num_lines] is added!!!
        idx_id = np.array([_i for _i in range(num_lines)
                if str_lines[_i][0] == id_tag] + [num_lines], dtype=np.int)

        # find the indices for real sequences only (>=2 lines for fasta, >=3 for seq)
        # e.g., multiple ";" lines are possible in SEQ format, only need the last one
        if len(idx_id) == 1:
            logger.critical(f'No lines with id tag: {id_tag} found!!!')
            idx_seqs = []
        else:
            # remove consecutive lines with the id tag (only applicable for SEQ format)
            nlines_per_seq = idx_id[1:] - idx_id[:-1]
            idx_seqs = np.where(nlines_per_seq > (2 if fmt == 'seq' else 1))[0]

        num_seqs = len(idx_seqs)
        logger.info(f'Found {len(idx_id) - 1} id tags, and {num_seqs} candidate sequences')

        # Group fasta_lines into groups for parallel processing
        lines_grpby_seq = []
        for i, iline in enumerate(idx_id[:-1]):
            # logger.debug(f'Processing sequence # {iseq}')
            if i not in idx_seqs: continue
            lines_grpby_seq.append(str_lines[iline:idx_id[i + 1]])

        # Ready to parse each sequence
        parse_func = functools.partial(parse_sequence_lines, fmt=fmt, id=id_tag, seq=seq,
                    dbn=dbn, upp=upp, return_dict=True)

        if num_seqs > 7:
            num_cpus = round(mpi.cpu_count() * 0.8)
            logger.info(f'Using multiprocessing for parsing, cpu count: {num_cpus}')
            mpool = mpi.Pool(processes=num_cpus)

            parsed_seq_dicts = mpool.map(parse_func, lines_grpby_seq)
            mpool.close()
        else:
            parsed_seq_dicts = [parse_func(_seq_lines) for _seq_lines in lines_grpby_seq]

        # Assign into the structure
        self.add_seq_from_dict(parsed_seq_dicts, istart=istart, fname=fnames)

        return

    def parse_ct_file(self, fnames, istart=None, **kwargs):
        """ use pandas.read_csv() which is written in C
        Each line in ct files has 6 columns:
        resnum resname resnum_before resnum_after resnum_ct/0 resnum
         """
        if type(fnames) not in (list, tuple): fnames = [fnames]

        num_seqs = len(fnames)

        parse_func = functools.partial(parse_ct_lines, is_file=True, return_dict=True)

        if num_seqs > 7:
            num_cpus = round(mpi.cpu_count() * 0.8)
            logger.info(f'Using multiprocessing for parsing, cpu count: {num_cpus}')
            mpool = mpi.Pool(processes=num_cpus)
            parsed_seq_dicts = mpool.map(parse_func, fnames)
            mpool.close()
        else:
            parsed_seq_dicts = [parse_func(_fname) for _fname in fnames]

        # Assign into the structure
        self.add_seq_from_dict(parsed_seq_dicts, istart=istart, fname=fnames)
        return

    def parse_bpseq_file(self, fname):
        logger.warning('Not yet implemented')
        pass

    def __str__(self):
        return f'num_seqs: {len(self.seq)}, min_len: {self.len.min()}, max_len: {self.len.max()}'

    # def pad_length(self, max_len):
    #     """ pad seqence, upp, and secondary structures to max_len """
    #     for i in range(self._len):
    #         if self.numResids[i] < max_len: # adding dummy residues
    #             if self.seq[i] is not None:
    #                 self.seq[i] += '-' * (max_len-self.numResids[i])
    #             if self.dbn[i] is not None:
    #                 self.dbn[i] += '-' * (max_len-self.numResids[i])
    #             if self.upp[i] is not None:
    #                 self.upp[i] = np.concatenate((self.upp[i],
    #                             np.ones((max_len - self.numResids[i],))), axis=0)
    #         elif self.numResids[i] == max_len:
    #             pass
    #         else: # removing residues from the end
    #             if self.seq[i] is not None:
    #                 self.seq[i] = self.seq[i][0:max_len]
    #             if self.dbn[i] is not None:
    #                 self.dbn[i] = self.dbn[i][0:max_len]
    #             if self.upp[i] is not None:
    #                 self.upp[i] = self.upp[i][0:max_len]

    @property
    def summary(self):
        """  """
        self.synopsis = dict(
            num_seq = len(self.seq),
            max_len = self.len.max(),
            min_len = self.len.min(),
            std_len = self.len.std()
        )
        return self.synopsis

    def to_df(self):
        database = pd.DataFrame(dict(
            idx = list(range(1, len(self.seq) + 1)),
            file = self.file, # [str(_f).replace(',', ':') for _f in self.file],
            id = self.id,
            len = self.len,
            seq = [_seq != '' for _seq in self.seq],
            dbn = [_dbn != '' for _dbn in self.dbn],
            upp = [len(_upp) > 0 for _upp in self.upp],
            ct = [len(_ct) > 0 for _ct in self.ct],
            numA = [_seq.count('A') for _seq in self.seq],
            numU = [_seq.count('U') for _seq in self.seq],
            numC = [_seq.count('C') for _seq in self.seq],
            numG = [_seq.count('G') for _seq in self.seq],
            upp_mean = [_upp.mean() if len(_upp) else None for _upp in self.upp],
            upp_std = [_upp.std() if len(_upp) else None for _upp in self.upp],
        ))

        return database

    def get_fasta_lines(self, idx=None, dbn=False, upp=False, **kwargs):

        if idx is None:
            idx = list(range(self.len))
        elif isinstance(idx, int) or isinstance(idx, np.integer):
            idx = [idx]

        fasta_lines = []
        for i in idx:
            fasta_lines.append('>' + self.id[i])
            fasta_lines.append(self.seq[i])
            if dbn: fasta_lines.append(self.dbn[i])
            if upp: fasta_lines.extend(self.upp[i].astype(str))

        return fasta_lines

    def write_sequence_file(self, fasta=None, dbn=None, bpseq=None, fdir='', id=None, single=True):
        assert bool(fasta) + bool(dbn) + bool(bpseq) == 1, \
            "One and only one of the 1D/2D structure files must be provided!"

        if fasta is not None:
            with open(os.path.join(fdir, fasta), 'w') as hfile:
                hfile.write(
                    '>' + (id if id is not None else 'unknown') + '\n')

                hfile.write(''.join(self.seq))
        elif dbn is not None:
            pass

    def write_base_pair_matrix(self, fname, fsuffix='.bpm', fdir=''):
        bpm_file = os.path.join(
            fdir, fname+fsuffix if '.' not in fname else fname)
        sav_mat = np.zeros(
            (self.len[0]+1, self.len[0]+1), dtype=int)
        sav_mat[1:, 1:] = self.bp_mat
        sav_mat[0, :] = np.linspace(
            0, self.len[0], num=self.len[0]+1, dtype=int)
        sav_mat[:, 0] = np.linspace(
            0, self.len[0], num=self.len[0]+1, dtype=int)
        sav_mat = sav_mat.astype(str)
        sav_mat[0, 0] = ''
        np.savetxt(bpm_file, sav_mat, fmt='%4s')
