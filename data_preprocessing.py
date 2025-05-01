import numpy as np
import pandas as pd
import tensorflow as tf
# from itertools import groupby

# Global variables
## Hyperparameters
windowsize = 145  
seqminsize = 149
paddings = tf.constant([[2, 2], [0, 0]])
aas = "ACDEFGHIKLMNPQRSTVWY"

def get_sequences_from_excel(input_file, sheets):
    """
    Extract sequences and labels from an Excel file with multiple sheets.
    Removes sequences containing 'X' or 'Z', truncates at '*', 
    and splits long sequences into fixed-size windows.
    
    Args:
        input_file (str): Path to the Excel file.
        sheets (list): List of sheet names to read from.
    Returns:
        tuple: (seqnames, sequences, labels)
    """

    # this is for the entire data in the excel file
    # seqdf_list = [pd.read_excel(input_file, sheet_name=sheet) for sheet in sheets]
    # seqdf = pd.concat(seqdf_list, ignore_index=True)
    # for testing purposes, we will only read the first 10 rows from the first sheet

    seqdf = pd.read_excel(input_file, sheet_name=sheets[0])
    sequences = seqdf.iloc[:200, :]  # Read only the first 200 rows for testing
    # Extract sequences and labels
    # sequences = seqdf.iloc[:, 0].tolist() 
    labels = seqdf.iloc[:200, 1].tolist() 
    clean_sequences, clean_labels = [], []

    for i, seq in enumerate(sequences):
        if "X" in seq or "Z" in seq:  # Skip sequences with invalid characters
            continue

        seq = seq.split('*')[0]  # Truncate at '*' means he is eliminating at the first * he finds
        # for example if
        # added the sup35n to the sequence
        seq="MDSTVSNFQKILTNPQNGGAQSNYGGGQSQGGYQQQNGGGQQQGGYQQQNYGPQGSWGQPHGGGWGQPHGGGWGQPHGGGWGQGGGTHSQWNKPSKPKTNMKHMAGAAAAGAVVGGLGGYMLGSAMSRPIIENF"+seq
        
        if len(seq) > windowsize:
            seq_chunks = [seq[j:j+windowsize] for j in range(0, len(seq), windowsize)] # this will make sure to add the sequence in chunks of 145
        else:
            seq_chunks = [seq]

        # clean_seqnames.extend([seqnames[i]] * len(seq_chunks))
        clean_sequences.extend(seq_chunks)
        clean_labels.extend([labels[i]] * len(seq_chunks)) 
    return  clean_sequences, clean_labels


def dynamic_padding(inp, min_size, consval=-1, post=True):
    pad_size = min_size - tf.shape(inp)[0]
    paddings = [[0, pad_size], [0, 0]] if post else [[pad_size, 0], [0, 0]] # this will add padding to the end of the sequence
    return tf.pad(inp, paddings, constant_values=consval)

def str_to_vector(str_seq, template):
    mapping = dict(zip(template, range(len(template))))
    
    seq = [mapping[i] for i in str_seq]
    return np.eye(len(template))[seq]

def seq_to_vector(seq):
    return str_to_vector(seq, aas)

def seq_to_embeds(seq):
    return dynamic_padding(tf.pad(seq_to_vector(seq), paddings, "CONSTANT"), seqminsize, post=True)

