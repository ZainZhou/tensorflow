import numpy as np

def load_data_and_labels(pos_path,neg_path):
    positive = open(pos_path,'rb').read().decode('utf-8')
    negative = open(neg_path,'rb').read().decode('utf-8')

    positive_examples = positive.split('\n')[:-1]
    negative_examples = negative.split('\n')[:-1]

    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]

    x_text = positive_examples + negative_examples
    # x_text = [clean_str(s) for s in x_text]

    positive_label = [[0,1] for _ in positive_examples]
    negative_label = [[1,0] for _ in negative_examples]
    y = np.concatenate([positive_label,negative_label],0)

    return [x_text,y]

def bathch_iter(data,batch_size,num_epoch,shuffle = True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)-1)/batch_size + 1
    for epoch in range(num_epoch):
        if shuffle:
            shuffle_index = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_index]
        else:
            shuffle_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num*batch_size
            end_index = min((batch_num+1)*batch_size,data_size)
            yield shuffle_data[start_index:end_index]

def clean_str(s):
    pass