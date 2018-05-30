# coding=utf-8
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import pickle
import os
import sys
import csv

import pandas as pd
import numpy as np
import tensorflow as tf

data_path = "/home/zyt/data/JD/test.csv"


def _build_vocab_from_url(filename):
    data = pd.read_csv(filename)

    request_url = data['request_url'].tolist()
    referer_url = data[data['referer_url'] != '-1']['referer_url'].tolist()
    urls = request_url + referer_url

    counter = collections.Counter(urls)
    count_pairs = sorted(counter.iteritems(), key=lambda x: (-x[1], x[0]))
    ranked_urls, _ = list(zip(*count_pairs))

    # 词表的大小为url的数目加上2, id=0表示未知词，id=1表示段结束
    url_to_id = dict(zip(ranked_urls, [i+2 for i in range(len(ranked_urls))]))
    url_to_id['UNK'] = 0
    url_to_id['<eos>'] = 1
    id_to_url = dict(zip([i+2 for i in range(len(ranked_urls))], ranked_urls))
    id_to_url[0] = 'UNK'
    id_to_url[1] = '<eos>'
    print("build vocabulary successfully!")
    output = open("data/processed_data/url_to_id.pkl", 'wb')
    pickle.dump(url_to_id, output)
    output.close()

    return url_to_id, id_to_url


def _path_to_ids(path, url_to_id):
    path_ids = list()
    for url in path:
        if url in url_to_id:
            path_ids.append(url_to_id[url])
        else:
            path_ids.append(url_to_id['UNK'])
    return path_ids


def _get_route_from_data(filename, url_to_id, min_path_len=3):
    routes = list()

    data = pd.read_csv(filename, chunksize=100000, encoding='utf-8')
    print("process data...")
    cur_ipto = 0   # current iptonumber
    path = []
    count = 0
    for chunk in data:
        print("process chunk_{}".format(count))
        sorted_chunk = chunk.sort_values(['iptonumber', 'visit_time'])
        filter_chunk = sorted_chunk[['visit_time', 'iptonumber', 'request_url', 'referer_url']]

        rows_iter = filter_chunk.itertuples()

        # row[0]: index, row[1]: visit_time, row[2]: iptonumber, row[3]: request_url, row[4]: referer_url

        for row in rows_iter:
            if row[2] != cur_ipto:
                if(len(path)>=min_path_len):
                    path_ids = _path_to_ids(path+['<eos>'], url_to_id)
                    map(lambda x: routes.append(x), path_ids)
                path = []
                request_url = row[3]
                referer_url = row[4]
                if(referer_url!='-1'):
                    path.append(referer_url)

                path.append(request_url)
                if(len(path)>0):
                    cur_ipto = row[2]
            else:
                request_url = row[3]
                referer_url = row[4]
                if (referer_url!='-1' and (referer_url!=path[len(path)-1] or len(path)==0)):
                    path.append(referer_url)
                path.append(request_url)

        count += 1

    if(len(path)>=min_path_len):
        path_ids = _path_to_ids(path+['<eos>'], url_to_id)
        map(lambda x: routes.append(x), path_ids)

    print("process data successfully!")

    return routes


def ptb_raw_data(data_path=None):
    """Load PTB raw data from data directory "data_path".

    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    The PTB dataset comes from Tomas Mikolov's webpage:

    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

    Args:
      data_path: string path to the directory where simple-examples.tgz has
        been extracted.

    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, "20160720.csv")
    valid_path = os.path.join(data_path, "20160727.csv")

    url_to_id, _ = _build_vocab_from_url(train_path)
    # 也可以从文件载入
    # input_file = open("data/processed_data/url_to_id.pkl", 'rb')
    # url_to_id = pickle.load(input_file)
    # input_file.close()
    datas = _get_route_from_data(train_path, url_to_id)
    train_data = datas[0:1000000]
    valid_data = datas[1000000:]
    test_data = _get_route_from_data(valid_path, url_to_id)
    vocabulary = len(url_to_id)
    return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
      name: the name of this operation (optional).

    Returns:
      A pair of Tensors, each shaped [batch_size, num_steps]. The second element
      of the tuple is the same data time-shifted to the right by one.

    Raises:
      tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y


def main(_):
    # for test
    filename = "/home/zyt/data/JD/20160720.csv"
    url_to_id, _ = _build_vocab_from_url(filename)
    routes = _get_route_from_data(filename,url_to_id)
    print(len(routes))

if __name__ == "__main__":
    tf.app.run()




