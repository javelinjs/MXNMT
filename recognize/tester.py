# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "../")
from nmt.xutils import load_vocab
from nmt.xutils import read_content, load_vocab, sentence2id, word2id
from inference import BiInferenceModel
from mxwrap.rnn.LSTM import lstm, LSTMModel, LSTMParam, LSTMState
import xconfig
from collections import namedtuple
import logging
import mxnet as mx
import numpy as np

BeamNode = namedtuple("BeamNode", ["father", "content", "score", "acc_score", "finishLen"])

def _smallest(matrix, k, only_first_row=False):
    """Find k smallest elements of a matrix.

    Parameters
    ----------
    matrix : :class:`numpy.ndarray`
        The matrix.
    k : int
        The number of smallest elements required.
    only_first_row : bool, optional
        Consider only elements of the first row.

    Returns
    -------
    Tuple of ((row numbers, column numbers), values).

    """
    if only_first_row:
        flatten = matrix[:1, :].flatten()
    else:
        flatten = matrix.flatten()
    args = np.argpartition(flatten, k)[:k]
    args = args[np.argsort(flatten[args])]
    return np.unravel_index(args, matrix.shape), flatten[args]

def recognize_one_with_beam(sentence, vocab, beam_size, ctx, arg_params):
    input_length = len(sentence)
    cur_model = BiInferenceModel(xconfig.num_lstm_layer,
                    len(sentence), len(vocab), xconfig.num_hidden,
                    xconfig.num_embed, xconfig.dropout,
                    arg_params, xconfig.target_label_size, ctx=ctx, batch_size=beam_size)
    input_raw = mx.nd.zeros((beam_size, input_length))
    input_raw[:] = np.array([sentence2id(sentence, vocab) for i in range(beam_size)])
    embeddings = cur_model.embed(input_raw, input_length)

    beam = [[BeamNode(father=-1, content=-1, score=0.0, acc_score=0.0, finishLen=0) for i in
             range(beam_size)]]
    beam_forward_state = [None]
    beam_backward_state = [None]

    for i in range(input_length):
        prob, new_forward_state, new_backward_state = cur_model.encode(
            embeddings[i], embeddings[input_length-1-i], beam_forward_state[-1], beam_backward_state[-1])
        log_prob = -mx.ndarray.log(prob)
        for idx in range(beam_size):
            log_prob[idx] = (log_prob[idx] + beam[-1][idx].acc_score * beam[-1][idx].finishLen) / (
                beam[-1][idx].finishLen + 1)

        (indexes, next_labels), chosen_costs = _smallest(log_prob.asnumpy(), beam_size, only_first_row=(i == 0))

        next_forward_state_h = mx.nd.empty(new_forward_state.h.shape, ctx=ctx)
        next_forward_state_c = mx.nd.empty(new_forward_state.c.shape, ctx=ctx)
        next_backward_state_h = mx.nd.empty(new_backward_state.h.shape, ctx=ctx)
        next_backward_state_c = mx.nd.empty(new_backward_state.c.shape, ctx=ctx)

        for idx in range(beam_size):
            next_forward_state_h[idx] = new_forward_state.h[np.asscalar(indexes[idx])]
            next_forward_state_c[idx] = new_forward_state.c[np.asscalar(indexes[idx])]
            next_backward_state_h[idx] = new_backward_state.h[np.asscalar(indexes[idx])]
            next_backward_state_c[idx] = new_backward_state.c[np.asscalar(indexes[idx])]

        next_forward_state = LSTMState(c=next_forward_state_c, h=next_forward_state_h)
        next_backward_state = LSTMState(c=next_backward_state_c, h=next_backward_state_h)

        beam_forward_state.append(next_forward_state)
        beam_backward_state.append(next_backward_state)

        next_beam = [BeamNode(father=indexes[idx],
                              content=next_labels[idx],
                              score=chosen_costs[idx] - beam[-1][indexes[idx]].acc_score,
                              acc_score=chosen_costs[idx],
                              finishLen=beam[-1][indexes[idx]].finishLen + 1) for idx in range(beam_size)]
        beam.append(next_beam)
    all_result = []
    all_score = []
    for aaa in range(beam_size):
        ptr = aaa
        result = []

        for idx in range(len(beam) - 1 - 1, 0, -1):
            label = beam[idx][ptr].content
            if label != -1:
                result.append(label)
            ptr = beam[idx][ptr].father
        result = result[::-1]
        all_result.append(result)
        all_score.append(beam[-1][aaa].acc_score)

    return all_result, all_score

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
        level=logging.INFO, datefmt='%H:%M:%S')
    # load vocabulary
    source_vocab = load_vocab(xconfig.source_vocab_path, xconfig.special_words)
    print('source_vocab size: {0}'.format(len(source_vocab)))
    # load model from check-point
    _, arg_params, __ = mx.model.load_checkpoint(xconfig.model_to_load_prefix, xconfig.model_to_load_number)
    with open(xconfig.test_input_file, mode='r', encoding='utf-8') as f:
        for line in f:
            all_result, all_score = recognize_one_with_beam(line.strip(), source_vocab,
                xconfig.test_beam_size, xconfig.test_device, arg_params)
            print(all_result)
            print(all_score) 
