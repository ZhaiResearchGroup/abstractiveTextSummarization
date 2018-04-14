def parse_batch(batch, use_cuda=True):
    input_batches = batch.story
    input_lengths = len(batch.story)
    query_batch = batch.raw_query
    query_length = len(batch.raw_query)
    sen_vec = batch.sen_vec.transpose(0,1)
    sen_idx = batch.sen_idx.transpose(0,1)
    target_batches = batch.sum
    target_lengths = len(batch.sum)
    if use_cuda:
        input_batches = input_batches.cuda()
        sen_vec = sen_vec.cuda()
        sen_idx = sen_idx.cuda()
        target_batches = target_batches.cuda()
        query_batch = query_batch.cuda()

    #TODO: check sentence_batches, sentence_lengths
    return input_batches, input_lengths, target_batches, target_lengths, sen_vec, sen_idx, query_batch
