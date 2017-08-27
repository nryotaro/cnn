# -*- coding: utf-8 -*-


def _count_txt_file_lines(src):

    def count(f):
        return sum(1 for line in f)

    if isinstance(src, str):
        with open(src, 'r') as f:
            return count(f)

    return count(src)


def create_batch_gen(src, batch_size):
    pass

"""
def create_batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
"""
