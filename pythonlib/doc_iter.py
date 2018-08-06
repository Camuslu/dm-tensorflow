# import simplejson as json
import json as json
import argparse
from collections import namedtuple
import numpy as np
import itertools as IT

FLAGS = None
LOC_LENGTH = 5

#TODO implement a filter mechanism s.t. document_iterator yields the result only if the line meets filter predicate

def document_iterator(file, query_keys = [], fns = []):
    """

    :param file: the file containing the train/test data, assuming each line is of json format (i.e. parsable as dict in Python)
    :param query_keys: helper list for query:
                        if all the values are in file[key1][key2], pass query_keys = [key1, key2]
    :param fns: list of functions to operate on file[query_keys]. fn(dict) should be of type list OR np.ndarray
    :return: iterator that yields tuple of (fn1(file), fn2(file),...)

    """
    with open(file) as f:
        for line in f:
            js = json.loads(line)
            for key in query_keys:
                js = js[key]
            yield [list(fn(js)) for fn in fns] #if fn returns np.array, convert it to list


def grouper(n, iterable):
    """
    >>> list(grouper(3, 'ABCDEFG'))
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]
    """
    iterable = iter(iterable)
    return iter(lambda: list(IT.islice(iterable, n)), [])


class BatchIterator:
    def __init__(self, iterator, batchsize, need_padding, pads):
        """

        :param iterator:
        :param batchsize:
        :param need_padding: list of Boolean.
                            e.g. [True, False, True] => feature[0] and feature[2] need padding.
        :param pads: value to pad to each feature that needs padding. e.g.:[0, None, 1000]
        """
        self.iterator = iterator
        self.batchsize = batchsize
        self.need_padding = need_padding
        self.num_feat = len(need_padding)
        self.pads = pads
        self.num_feat_with_pad = sum(self.need_padding)

    def get(self):
        for el in grouper(self.batchsize, self.iterator):
            bsize = len(el)

            seq_lengths = [0] * self.num_feat # for current batch, the seq length for each feature that need padding
            #el is [tuple_1, tuple_2, ...tuple_bsize]

            iter_result = [[[]]*bsize for _ in range(self.num_feat)]


            # First fill the iter_result with raw features from el
            for (idx1,feat_tuple) in enumerate(el):
                for (idx2, feat) in enumerate(feat_tuple):
                    iter_result[idx2][idx1] = feat
                    seq_lengths[idx2] = max(seq_lengths[idx2], len(feat))

            # Then do padding based on updated seq_lengths
            for (idx2, batched_feat) in enumerate(iter_result):
                if self.need_padding[idx2] == True:
                    for (idx1, feat) in enumerate(batched_feat):
                        feat_padded = feat + [self.pads[idx2]]*(seq_lengths[idx2] - len(feat))
                        iter_result[idx2][idx1] = feat_padded
            # print(iter_result)
            yield tuple([np.array(batch) for batch in iter_result])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainData', type=str, help='Input train data')
    parser.add_argument('--testData', type=str, help='Input test data')
    parser.add_argument('--classes', type=str, help='File with list of classes')
    parser.add_argument('--embedSize', type=int, help='Embedding size')
    parser.add_argument('--seqLength', type=int, help='Max length of a sequence')
    parser.add_argument('--batchSize', type=int, help='Size of 1 batch')

    FLAGS, unparsed = parser.parse_known_args()
    iterator = document_iterator(FLAGS.trainData, FLAGS.embedSize, FLAGS.seqLength, FLAGS.classes)
    batch_iterator = BatchIterator(iterator, FLAGS.batchSize)
    for _ in range(10000):
        for e in batch_iterator.get():
            print( )



