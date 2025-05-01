import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation, ActivityRegularization, BatchNormalization, Conv1D, 
    Dense, Dropout, Flatten, Layer, Multiply, Masking, MaxPool1D
)
from tensorflow.keras.regularizers import L1L2
import numpy as np
from pkg_resources import resource_filename

# Constants
SEQ_MIN_SIZE = 149
PADDINGS = tf.constant([[2, 2], [0, 0]])
AAS = "ACDEFGHIKLMNPQRSTVWY"

def dynamic_padding(inp, min_size, consval=-1, post=True):
    """Dynamically pad sequences to reach minimum size"""
    pad_size = min_size - tf.shape(inp)[0]
    paddings = [[0, pad_size], [0, 0]] if post else [[pad_size, 0], [0, 0]]
    return tf.pad(inp, paddings, constant_values=consval)

def str_to_vector(s, template):
    """Convert string sequence to one-hot vector"""
    mapping = dict(zip(template, range(len(template))))
    seq = [mapping[i] for i in s]
    return np.eye(len(template))[seq]

def seq_to_vector(seq):
    """Convert protein sequence to one-hot encoding"""
    return str_to_vector(seq, AAS)

def binary_KL(y_true, y_pred):
    """Custom KL divergence loss function for TF 2.6.1"""
    return tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, y_pred) - 
        tf.keras.losses.binary_crossentropy(y_true, y_true),
        axis=-1
    )

class GetAttentionMask(Layer):
    """Create attention mask from sequence mask (TF 2.6.1 compatible)"""
    def __init__(self, num_repeats, **kwargs):
        super(GetAttentionMask, self).__init__(**kwargs)
        self.num_repeats = num_repeats
    
    def call(self, inputs):
        mask = tf.squeeze(inputs)
        mask = tf.abs(tf.subtract(inputs, 1.0))
        mask = tf.repeat(mask, repeats=mask.shape[-2], axis=-1)
        mask = tf.transpose(mask, [0, 2, 1])
        return mask[:, tf.newaxis, :, :]
    
    def get_config(self):
        config = super(GetAttentionMask, self).get_config()
        config.update({'num_repeats': self.num_repeats})
        return config

class MultiHeadAttentionWithPosition(Layer):
    """Multi-head attention with positional encoding for TF 2.6.1"""
    def __init__(self, d_model, num_heads, embedding_size=None, **kwargs):
        super(MultiHeadAttentionWithPosition, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.embedding_size = d_model if embedding_size is None else embedding_size
        self.depth = d_model // num_heads

        self.wq = Dense(d_model, use_bias=False)
        self.wk = Dense(d_model, use_bias=False)
        self.wv = Dense(d_model, use_bias=False)
        self.r_k_layer = Dense(d_model, use_bias=False)
        self.dense = Dense(d_model)
        self.dropout = Dropout(0.1)

    def build(self, input_shape):
        self.r_w = self.add_weight(
            name='r_w',
            shape=[1, self.num_heads, 1, self.depth],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),
            trainable=True
        )
        self.r_r = self.add_weight(
            name='r_r',
            shape=[1, self.num_heads, 1, self.depth],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),
            trainable=True
        )
        super().build(input_shape)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, [0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)
        
        q = q / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        seq_len = tf.shape(q)[2]
        pos = tf.range(-seq_len + 1, seq_len, dtype=tf.float32)[tf.newaxis]
        feature_size = self.embedding_size // 6
        
        # Positional encodings (compatible with TF 2.6.1)
        exp1 = tf.exp(-tf.math.log(2.0) / (2.0 ** tf.linspace(3.0, 10.0, feature_size)) * tf.abs(pos)[..., tf.newaxis])
        cm1 = tf.cast((2.0 ** tf.range(1, feature_size+1, dtype=tf.float32) - 1 > tf.abs(pos)[..., tf.newaxis], tf.float32))
        gam = tf.math.exp(
            (tf.linspace(1.0, 10.0, feature_size) - 1) * tf.math.log(tf.abs(pos)[..., tf.newaxis]) - 
            tf.linspace(1.0, 10.0, feature_size) * tf.abs(pos)[..., tf.newaxis] - 
            (tf.math.lgamma(tf.linspace(1.0, 10.0, feature_size)) - 
             tf.linspace(1.0, 10.0, feature_size) * tf.math.log(tf.linspace(1.0, 10.0, feature_size)))
        )
        
        pos_enc = tf.concat([exp1, exp1 * tf.sign(pos)[..., tf.newaxis], 
                            cm1, cm1 * tf.sign(pos)[..., tf.newaxis],
                            gam, gam * tf.sign(pos)[..., tf.newaxis]], axis=-1)
        pos_enc = self.dropout(pos_enc)
        
        r_k = tf.transpose(
            tf.reshape(self.r_k_layer(pos_enc), [1, -1, self.num_heads, self.depth]),
            [0, 2, 1, 3]
        )

        content_logits = tf.matmul(q + self.r_w, k, transpose_b=True)
        relative_logits = tf.matmul(q + self.r_r, r_k, transpose_b=True)
        
        # Relative shift implementation for TF 2.6.1
        pad = tf.zeros_like(relative_logits[..., :1])
        relative_logits = tf.concat([pad, relative_logits], -1)
        relative_logits = tf.reshape(relative_logits, [-1, *relative_logits.shape[1:-2], relative_logits.shape[-1], relative_logits.shape[-2]])
        relative_logits = relative_logits[..., 1:, :]
        relative_logits = tf.reshape(relative_logits, [-1, *relative_logits.shape[1:-2], relative_logits.shape[-2], relative_logits.shape[-1]-1])[..., :(relative_logits.shape[-1]+1)//2]
        
        logits = content_logits + relative_logits
        logits = tf.where(tf.equal(mask, 1), -1e9, logits)
        attention = tf.nn.softmax(logits)

        output = tf.matmul(attention, v)
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        return self.dense(output), attention

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'embedding_size': self.embedding_size
        })
        return config

def create_model(
    indim=20,
    input_length=149,
    dense_number=64,
    num_filters=100,
    filter_width=3,
    dropout_rate=0.4,
    num_heads=1,
    key_len=6,
    l2_regularization=0.01,
    pool_size=None,
    batch_size=256,
    act_reg=0,
    **kwargs
):
    """Create CANYA model architecture for TF 2.6.1"""
    seq_input = Input(shape=(input_length, indim), name='seq_input')
    
    # Masking
    masked_input = Masking(mask_value=-1)(seq_input)
    mask = tf.cast(Masking(mask_value=-1).compute_mask(seq_input), tf.float32)
    
    # Attention mask
    if pool_size:
        mask = MaxPool1D(pool_size=pool_size)(mask[:, :, tf.newaxis])
    attn_mask = GetAttentionMask(mask.shape[1] if pool_size else input_length)(mask[:, :, tf.newaxis])

    # Convolutional block
    conv = Conv1D(num_filters, filter_width, padding='same', use_bias=False)(masked_input)
    conv = Activation('exponential')(conv)
    conv = ActivityRegularization(l1=act_reg)(conv)
    conv = Multiply()([conv, mask[:, :, tf.newaxis]])
    
    if pool_size:
        conv = MaxPool1D(pool_size=pool_size)(conv)
    conv = Dropout(0.1)(conv)

    # Attention block
    attn, _ = MultiHeadAttentionWithPosition(
        num_heads=num_heads,
        d_model=num_heads*key_len
    )(conv, conv, conv, attn_mask)
    attn = Dropout(0.1)(attn)
    flat = Flatten()(attn)

    # Output block
    dense = Dense(dense_number, kernel_regularizer=L1L2(l2_regularization))(flat)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dense = Dropout(dropout_rate)(dense)
    
    output = Dense(1, activation='sigmoid')(dense)
    
    return Model(inputs=seq_input, outputs=output)

def get_custom_objects():
    """Return all custom objects needed for model loading in TF 2.6.1"""
    return {
        'binary_KL': binary_KL,
        'GetAttentionMask': GetAttentionMask,
        'MultiHeadAttentionWithPosition': MultiHeadAttentionWithPosition
    }

def get_canya(modweights="models/model_weights1.h5"):
    """Load pre-trained CANYA model in TF 2.6.1"""
    model = create_model()
    model.load_weights(resource_filename(__name__, modweights))
    return model

def get_embedded_seqs(sequences):
    """Convert sequences to padded embeddings in TF 2.6.1"""
    embedded = [tf.pad(seq_to_vector(x), PADDINGS) for x in sequences]
    embedded = [dynamic_padding(x, SEQ_MIN_SIZE) for x in embedded]
    return np.array(embedded)

def get_predictions(model, sequences):
    """Get predictions from model in TF 2.6.1"""
    return model.predict(sequences).flatten().tolist()