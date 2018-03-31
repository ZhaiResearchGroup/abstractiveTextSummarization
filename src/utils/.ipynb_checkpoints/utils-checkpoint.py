def parse_batch(batch, USE_CUDA):
    input_batches = batch.story
    input_lengths = len(batch.story)
    sen_vec = batch.sen_vec.transpose(0,1)
    sen_idx = batch.sen_idx.transpose(0,1)
    target_batches = batch.sum
    target_lengths = len(batch.sum)
    if USE_CUDA:
        input_batches = input_batches.cuda()
        sen_vec = sen_vec.cuda()
        sen_idx = sen_idx.cuda()
        target_batches = target_batches.cuda()

    #TODO: check sentence_batches, sentence_lengths
    return input_batches, input_lengths, target_batches, target_lengths, sen_vec, sen_idx

    #print('input_batches', input_batches.size()) # (max_len x batch_size)
    #print('sen_vec', sen_vec.size()) # (batch_size x max_num_sen x vec_len)
    #print('sen_idx', sen_idx.size()) # (batch_size x max_len)
    #print('target_batches', target_batches.size()) # (max_len x batch_size)
