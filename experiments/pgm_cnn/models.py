import sys; sys.path += ['..', '../..']
from autoregressive_abstractor import AutoregressiveAbstractor
from seq2seq_abstracter_models import Transformer
from abstractor import Abstractor
from transformer_modules import Encoder
import tensorflow as tf
from tensorflow.keras import layers, Model
#region common kwargs
d_model = 512
num_heads = 8
dff = 2048
num_layers = 1

def get_params_by_size(size):
    if size=='small':
        d_model, num_heads, dff, num_layers = (64, 2, 128, 1)
    elif size=='medium':
        d_model, num_heads, dff, num_layers = (128, 4, 256, 1)
    elif size=='medium+':
        d_model, num_heads, dff, num_layers = (200, 4, 400, 1)
    elif size=='medium++':
        d_model, num_heads, dff, num_layers = (256, 4, 512, 1)
    elif size=='large':
        d_model, num_heads, dff, num_layers = (256, 8, 1024, 2)
    elif size=='x-large':
        d_model, num_heads, dff, num_layers = (512, 8, 2048, 2)
    else:
        raise ValueError(f'size {size} invalid')

    return d_model, num_heads, dff, num_layers
#endregion


## Can do:
## 1. Abstractor only - CNN outputs (seq, d_model), feeds directly into Abstractor, which then outputs (seq, d_model), which then must be flattened, passed through final layer to give (d_ans = 8) logits
## 2. Abstractor + Decoder - CNN outputs, feeds into Abstractor, which then is decoded along with the CNN representation itself.
## 2. Abstractor + Decoder (with Enc input) - two inputs to decoder here.
## Can try to add positional embeddings to the symbols here.




class AbstractorCNNModel(tf.keras.Model):
    def __init__(self, abstractor_kwargs, encoder_kwargs, name=None):
        super().__init__(name=name)
        self.cnn = Encoder(encoder_type="cnn", **encoder_kwargs) 
        #num_layers, num_kernels, kernel_size, stride, input_size, dropout_rate=0.1, name="cnn_encoder", dropout_in_cnn=False, mlp=[], **kwargs
        self.abstractor = Abstractor(**abstractor_kwargs)
        #self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.flatten = layers.Flatten()
        # initialize decoder
        #self.decoder = MultiAttentionDecoder(**decoder_kwargs, name='decoder')
        self.hidden_dense = layers.Dense(32, activation='relu', name='hidden_layer')
        self.final_layer = layers.Dense(8, activation='sigmoid', name='final_layer')

    def call(self, inputs):
        ## Need to reshape (batch, seq_len, height, width) to (batch*seq_len, height,width, 1)
        seq_len = inputs.shape[1]
        assert seq_len == 16
        inputs = tf.reshape(inputs, [-1, inputs.shape[2], inputs.shape[3], 1])
        cnn_embeddings = self.cnn(inputs)
        x = tf.reshape(cnn_embeddings, [-1, seq_len, cnn_embeddings.shape[1]])
        x = self.abstractor(x) ## output is shape (batch, seq_len, d_model)
        x = self.flatten(x)
        x = self.hidden_dense(x)
        logits = self.final_layer(x)
        return logits

    def print_summary(self, input_shape):
        inputs = layers.Input(input_shape)
        outputs = self.call(inputs)
        print(tf.keras.Model(inputs, outputs, name=self.name).summary())



#region Transformer
def create_transformer(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)
    transformer = Transformer(
        num_layers=num_layers, num_heads=num_heads, dff=dff, embedding_dim=d_model,
        input_vocab=input_vocab_size, target_vocab=target_vocab_size,
        output_dim=target_vocab_size, dropout_rate=0.1,)

    return transformer
#endregion


#region Abstractor ('simple' implementation without TF's MHA)
def create_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)
    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=num_layers, rel_dim=num_heads, dff=dff, symbol_dim=d_model,
        proj_dim=d_model//num_heads, symmetric_rels=False, encoder_kwargs=None,
        rel_activation_type='softmax', use_self_attn=False, use_layer_norm=False,
        dropout_rate=0.1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='abstractor', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion


#region RelationalAbstractor (implementation with forked+adjusted version of TF's MHA)
def create_relational_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff,
        use_learned_symbols=False, mha_activation_type='softmax', use_self_attn=True)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='relational', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region RelationalAbstractor with linear relational activation
def create_linear_relational_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff,
        use_learned_symbols=False, mha_activation_type='linear', use_self_attn=False)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='relational', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region RelationalAbstractor with architecture c (abstractor on input, decoder on encoder, abstractor; see paper)
def create_relational_abstractor_archc(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff,
        use_learned_symbols=False, mha_activation_type='softmax', use_self_attn=True)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='relational', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='input', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region SymbolRetrievingAbstractor (abstractor with symbol--retrieval via symbolic attention)
def create_symbolretrieving_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(
        num_layers=num_layers, num_heads=num_heads, dff=dff,
        n_symbols=256, symbol_n_heads=num_heads, symbol_retriever_type=1,
        rel_activation_function='softmax', use_self_attn=True, dropout_rate=0.1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='symbol-retrieving', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='input', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region SyntacticAbstractor (an experimental variant)
def create_syntactic_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=1, num_heads=num_heads, dff=dff, n_symbols=16,
        symbol_n_heads=1, symbol_binding_dim=None, add_pos_embedding=True, symbol_retriever_type=1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='syntactic',
        abstractor_on='input', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')

    return abstractor
#endregion

#region SymbolRetrievingAbstractor with architecture d (abstractor on encoder, decoder on encoder-abstractor; see paper)
def create_symbolretrieving_abstractor_archd(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(
        num_layers=num_layers, num_heads=num_heads, dff=dff,
        n_symbols=256, symbol_n_heads=num_heads, add_pos_embedding=True, symbol_retriever_type=1,
        rel_activation_function='softmax', use_self_attn=True, dropout_rate=0.1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='symbol-retrieving', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region SymbolRetrievingAbstractor with architecture d and only a few symbols
def create_symbolretrieving_abstractor_archd_fewsymbols(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(
        num_layers=num_layers, num_heads=num_heads, dff=dff,
        n_symbols=16, symbol_n_heads=1, add_pos_embedding=True, symbol_retriever_type=1,
        rel_activation_function='softmax', use_self_attn=True, dropout_rate=0.1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='symbol-retrieving', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion


#region SymbolRetrievingAbstractor with a single-head symbolic attention
def create_symbolretrieving_singlehead_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(
        num_layers=num_layers, num_heads=num_heads, dff=dff,
        n_symbols=256, symbol_n_heads=1, symbol_retriever_type=1,
        rel_activation_function='softmax', use_self_attn=True, dropout_rate=0.1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='symbol-retrieving', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='input', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion


model_creator_dict = dict(
    transformer=create_transformer,
    abstractor=create_abstractor,
    relational_abstractor=create_relational_abstractor,
    syntactic_abstractor=create_syntactic_abstractor,
    symbolretrieving_abstractor=create_symbolretrieving_abstractor,
    symbolretrieving_singlehead_abstractor=create_symbolretrieving_singlehead_abstractor,
    symbolretrieving_abstractor_archd=create_symbolretrieving_abstractor_archd,
    symbolretrieving_abstractor_archd_fewsymbols=create_symbolretrieving_abstractor_archd_fewsymbols,
    linear_relational_abstractor=create_linear_relational_abstractor,
    relational_abstractor_archc=create_relational_abstractor_archc
    )
