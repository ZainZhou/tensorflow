import numpy as np

def load_data_and_labels(pos_path,neg_path):
    positive = open(pos_path,'rb').read().decode('utf-8')
    negative = open(neg_path,'rb').read().decode('utf-8')

    positive_examples = positive.split('\n')[:-1]
    negative_examples = negative.split('\n')[:-1]

    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]

    x_text = positive_examples + negative_examples
    x_text = [clean_str(s) for s in x_text]

    positive_label = [[0,1] for _ in positive_examples]
    negative_label = [[1,0] for _ in negative_examples]
    y = np.concatenate([positive_label,negative_label],0)

    return [x_text,y]

def clean_str(s):
    pass