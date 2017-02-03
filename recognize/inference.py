# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "../")

from mxwrap.seq2seq.encoder import BiDirectionalLstmEncoder
from mxwrap.rnn.LSTM import lstm, LSTMModel, LSTMParam, LSTMState
from nmt.xutils import read_content, load_vocab, sentence2id, word2id
import mxnet as mx
import numpy as np

def embedding_symbol(num_embed, seq_len, vocab_size):
    data = mx.sym.Variable('source')  # input data, source
    # declare variables
    embed_weight = mx.sym.Variable("source_embed_weight")
    # embedding layer
    # FIXME: should be + 1
    embed = mx.sym.Embedding(data=data, input_dim=vocab_size + 2,
                             weight=embed_weight, output_dim=num_embed, name='source_embed')
    wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)
    output = []
    for i in range(seq_len):
        output.append(wordvec[i])
    return mx.sym.Group(output)

def bidirectional_encode_symbol(num_of_layer, vocab_size, num_hidden, num_embed, dropout, target_size):
    forward_param_cells = []
    forward_last_states = []
    for i in range(num_of_layer):
        forward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("forward_source_l%d_i2h_weight" % i),
                                             i2h_bias=mx.sym.Variable("forward_source_l%d_i2h_bias" % i),
                                             h2h_weight=mx.sym.Variable("forward_source_l%d_h2h_weight" % i),
                                             h2h_bias=mx.sym.Variable("forward_source_l%d_h2h_bias" % i)))
        forward_state = LSTMState(c=mx.sym.Variable("forward_source_l%d_init_c" % i),
                                  h=mx.sym.Variable("forward_source_l%d_init_h" % i))
        forward_last_states.append(forward_state)
    assert (len(forward_last_states) == num_of_layer)
    backward_param_cells = []
    backward_last_states = []
    for i in range(num_of_layer):
        backward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("backward_source_l%d_i2h_weight" % i),
                                              i2h_bias=mx.sym.Variable("backward_source_l%d_i2h_bias" % i),
                                              h2h_weight=mx.sym.Variable("backward_source_l%d_h2h_weight" % i),
                                              h2h_bias=mx.sym.Variable("backward_source_l%d_h2h_bias" % i)))
        backward_state = LSTMState(c=mx.sym.Variable("backward_source_l%d_init_c" % i),
                                   h=mx.sym.Variable("backward_source_l%d_init_h" % i))
        backward_last_states.append(backward_state)
    assert (len(backward_last_states) == num_of_layer)

    forward_hidden = mx.sym.Variable("forward_hidden_input")
    backward_hidden = mx.sym.Variable("backward_hidden_input")

    # stack LSTM
    for i in range(num_of_layer):
        if i == 0:
            dp_ratio = 0.
        else:
            dp_ratio = dropout
        forward_next_state = lstm(num_hidden, indata=forward_hidden,
                                  prev_state=forward_last_states[i],
                                  param=forward_param_cells[i],
                                  seqidx=0, layeridx=i, dropout=dp_ratio)
        backward_next_state = lstm(num_hidden, indata=backward_hidden,
                                   prev_state=backward_last_states[i],
                                   param=backward_param_cells[i],
                                   seqidx=0, layeridx=i, dropout=dp_ratio)

        forward_hidden = forward_next_state.h
        forward_last_states[i] = forward_next_state
        backward_hidden = backward_next_state.h
        backward_last_states[i] = backward_next_state

    if dropout > 0.:
        forward_hidden = mx.sym.Dropout(data=forward_hidden, p=dropout)
        backward_hidden = mx.sym.Dropout(data=backward_hidden, p=dropout)

    bi = mx.sym.Concat(forward_hidden, backward_hidden, dim=1)

    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    pred = mx.sym.FullyConnected(data=bi, num_hidden=target_size,
                                 weight=cls_weight, bias=cls_bias, name='target_pred')

    sm = mx.sym.SoftmaxOutput(data=pred, name='target_softmax')
    output = [sm,
        forward_last_states[-1].c, forward_last_states[-1].h,
        backward_last_states[-1].c, backward_last_states[-1].h]
    return mx.sym.Group(output)

class BiInferenceModel(object):
    def __init__(self, num_lstm_layer, seq_len, vocab_size, num_hidden, num_embed, dropout,
                 arg_params, target_size, ctx=mx.cpu(), batch_size=1):
        self.embed_sym = embedding_symbol(num_embed, seq_len, vocab_size)
        self.encode_sym = bidirectional_encode_symbol(num_lstm_layer,
                                                      vocab_size, num_hidden, num_embed,
                                                      dropout, target_size)
        # initialize states for LSTM
        forward_source_init_c = [('forward_source_l%d_init_c' % l, (batch_size, num_hidden)) for l in
                                 range(num_lstm_layer)]
        forward_source_init_h = [('forward_source_l%d_init_h' % l, (batch_size, num_hidden)) for l in
                                 range(num_lstm_layer)]
        backward_source_init_c = [('backward_source_l%d_init_c' % l, (batch_size, num_hidden)) for l in
                                  range(num_lstm_layer)]
        backward_source_init_h = [('backward_source_l%d_init_h' % l, (batch_size, num_hidden)) for l in
                                  range(num_lstm_layer)]
        hidden_inputs = [('forward_hidden_input', (batch_size, num_embed)), ('backward_hidden_input', (batch_size, num_embed))]
        source_init_states = forward_source_init_c + forward_source_init_h \
          + backward_source_init_c + backward_source_init_h + hidden_inputs

        embed_input_shapes = {"source": (batch_size, seq_len)}
        self.embed_executor = self.embed_sym.simple_bind(ctx=ctx, grad_req='null', **embed_input_shapes)

        encode_input_shapes = dict(source_init_states)
        self.encode_executor = self.encode_sym.simple_bind(ctx=ctx, grad_req='null', **encode_input_shapes)

        for key in self.embed_executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.embed_executor.arg_dict[key])
        for key in self.encode_executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.encode_executor.arg_dict[key])

        encode_state_name = []
        for i in range(num_lstm_layer):
            encode_state_name.append("forward_source_l%d_init_c" % i)
            encode_state_name.append("forward_source_l%d_init_h" % i)
            encode_state_name.append("backward_source_l%d_init_c" % i)
            encode_state_name.append("backward_source_l%d_init_h" % i)

        self.encode_states_dict = dict(zip(encode_state_name, self.encode_executor.outputs[1:]))

    def embed(self, input_data, seq_len):
        input_data.copyto(self.embed_executor.arg_dict["source"])
        self.embed_executor.forward()
        embeddings = []
        for i in range(seq_len):
            embeddings.append(self.embed_executor.outputs[i])
        return embeddings

    def encode(self, forward_input_data, backward_input_data, forward_state, backward_state):
        for key in self.encode_states_dict.keys():
            self.encode_executor.arg_dict[key][:] = 0.
        forward_input_data.copyto(self.encode_executor.arg_dict["forward_hidden_input"])
        backward_input_data.copyto(self.encode_executor.arg_dict["backward_hidden_input"])
        if forward_state is not None:
            forward_state.c.copyto(self.encode_executor.arg_dict["forward_source_l0_init_c"])
            forward_state.h.copyto(self.encode_executor.arg_dict["forward_source_l0_init_h"])
        if backward_state is not None:
            backward_state.c.copyto(self.encode_executor.arg_dict["backward_source_l0_init_c"])
            backward_state.h.copyto(self.encode_executor.arg_dict["backward_source_l0_init_h"])
        self.encode_executor.forward()
        prob = self.encode_executor.outputs[0]
        forward_c = self.encode_executor.outputs[1]
        forward_h = self.encode_executor.outputs[2]
        backward_c = self.encode_executor.outputs[3]
        backward_h = self.encode_executor.outputs[4]
        return prob, LSTMState(c=forward_c, h=forward_h), LSTMState(c=backward_c, h=backward_h)
