import sys
sys.path.insert(0, "../")
from mxwrap.seq2seq.encoder import BiDirectionalLstmEncoder

import xconfig
import mxnet as mx

def bi_lstm_unroll(num_lstm_layer, seq_len, vocab_size, num_hidden, num_embed, dropout, target_size):
    encoder = BiDirectionalLstmEncoder(seq_len=seq_len, use_masking=True, state_dim=num_hidden,
                                       input_dim=vocab_size, output_dim=0,
                                       vocab_size=vocab_size, embed_dim=num_embed,
                                       dropout=dropout, num_of_layer=num_lstm_layer)
    label = mx.sym.Variable('softmax_label')
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")

    forward_hidden_all, backward_hidden_all, source_representations, source_mask_sliced = encoder.encode()
    hidden_concat = mx.sym.Concat(*source_representations, dim=0)

    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=target_size,
                                 weight=cls_weight, bias=cls_bias, name='target_pred')

    label = mx.sym.transpose(data=label)
    label = mx.sym.Reshape(data=label, shape=(-1,))

    # use mask
    loss_mask = mx.sym.transpose(data=encoder.input_mask)
    loss_mask = mx.sym.Reshape(data=loss_mask, shape=(-1, 1))
    pred = mx.sym.broadcast_mul(pred, loss_mask)

    sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='target_softmax')
    return sm

def sym_gen(source_vocab_size, label_size):
    def _sym_gen(seq_len):
        return bi_lstm_unroll(num_lstm_layer=xconfig.num_lstm_layer, seq_len=seq_len,
                          vocab_size=source_vocab_size + 1,
                          num_hidden=xconfig.num_hidden, num_embed=xconfig.num_embed,
                          dropout=xconfig.dropout,
                          target_size=label_size)
    return _sym_gen
