# coding=utf-8

from __future__ import print_function

import sys
import threading
import pickle
import re

from grpc.beta import implementations
import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


tf.app.flags.DEFINE_string('server', 'localhost:6007',
                           'predictionService host:port')

tf.app.flags.DEFINE_integer('batch_size', 1, '')
tf.app.flags.DEFINE_integer('num_steps', 5, '')

FLAGS = tf.app.flags.FLAGS


def test_search(url):
        # Search Engines Reg
        SousouReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*sousou\.com'
        SouGouReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*sogou\.com'
        Searcg360Reg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*so\.360\.cn/'
        BaiduReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*baidu\.com'
        BingReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*bing\.com'
        AolReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*aol\.com'
        AskReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*ask\.com'
        DaumReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*daum\.net'
        GoogleReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*google\.'
        MailReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*go\.mail\.ru'
        WebCrawlerReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*webcrawler\.com'
        WowReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*us\.wow\.com'
        YahooReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*yahoo\.(com|co)'
        YandexReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*yandex\.(com|by)'
        MySearchReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*zxyt\.cn'
        BingIEReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*bing\.ie'
        SearchLockReg = '^www\.searchlock\.com'
        SoSoReg = '^www\.soso\.com'
        SoReg = '^www\.so\.com'
        GoogleWebLightReg = '^googleweblight\.com'
        result = re.search(SouGouReg + '|' + SousouReg + '|' +Searcg360Reg + '|' + BaiduReg + '|' + BingReg +'|' + AolReg +'|' + AskReg +'|' + DaumReg +'|' +
                             GoogleReg +'|' + MailReg +'|' + WebCrawlerReg +'|' + WowReg +'|' + YahooReg +'|' + YandexReg +'|' + MySearchReg +'|' + BingIEReg +'|' +
                             SearchLockReg +'|' + SoSoReg +'|' + SoReg +'|' + GoogleWebLightReg, url)
        if result:
            return result.group()
        else:
            return None


def _filter_url(url):
        # Unify all external Search engines to External.Search.Engines
        InquirySuccess = r'^www\.made-in-china\.com/sendInquiry/success'
        InquiryFailure = r'^www\.made-in-china\.com/sendInquiry/failure'
        InquiryProd_ = r'^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*made-in-china\.com/(sendInquiry|sendinquiry)/prod_'

        # Multi-search Pages Reg
        multisearch = r'^www\.made-in-china\.com/multi-search/.*/F[0-2]/'
        # multisearchCatalog = r'^www\.made-in-china.com/[^/]*-Catalog/'
        t = test_search(url)
        if t is not None:
            term = t

        # Unify all sendInquiry/success pages  to sendInquiry/success
        elif re.search(InquirySuccess, url) is not None:
            term = r'www.made-in-china.com/sendInquiry/success'

        # Unify all sendInquiry/failure pages to sendInquiry/failure
        elif re.search(InquiryFailure, url) is not None:
            term = r'www.made-in-china.com/sendInquiry/failure'

        # Unify all senInquiry/prods_ pages tp sendInqiry/prod_
        elif re.search(InquiryProd_, url) is not None:
            term = r'www.made-in-china.com/sendInquiry/prods_item.html'

        # Unify all multi-search/.*F1 pages to www\.made-in-china\.multi-search/item/F1/pages\.html
        elif re.search(multisearch, url) is not None:
            term = r'www.made-in-china.com/multi-search/items/F1/pages.html'

        # # Unify all www\.made-in-china.com/multi-search/.*-Catalog/F0 pages to www.made-in-china.com/multi-search/items-Catalog/F0.html
        # elif re.search(multisearchCatalog, url) is not None:
        #     term = r'www.made-in-china.com/multi-search/items-Catalog/F0/pages.html'
        else:
            term = url

        if re.search(r'(((&|\?)(key)?word=)|(^membercenter\.))', url) is not None:
            term = r'www.made-in-china.com/membercenter.html'

        return term


def _path_to_ids(path, url_to_id):
    path_ids = list()
    for url in path:
        if url in url_to_id:
            path_ids.append(url_to_id[url])
        else:
            path_ids.append(url_to_id['UNK'])
    return path_ids


def inference(data):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    input_file = open("static/vocabulary/url_to_id.pkl", 'rb')
    url_to_id = pickle.load(input_file)
    input_file.close()

    vocab_size = len(url_to_id)
    delta = ['<eos>'] * FLAGS.num_steps
    raw_path = data.split(",")
    filtered_path = [_filter_url(url) for url in raw_path]
    path = filtered_path + delta

    path = path[:FLAGS.num_steps]
    output_len = FLAGS.num_steps
    if len(raw_path) < FLAGS.num_steps:
        output_len = len(raw_path)
    path_ids = _path_to_ids(path, url_to_id)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'ptb_serving'
    request.model_spec.signature_name = 'predict_signature'

    request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(path_ids, shape=[FLAGS.batch_size, FLAGS.num_steps]))

    result = stub.Predict(request, 10.0)

    outputs = result.outputs['output'].float_val
    output = outputs[(output_len-1)*vocab_size: output_len*vocab_size]
    path_representation = result.outputs['cell_state'].float_val
    embed_lookup = result.outputs['embed_lookup'].float_val

    return output, path_representation, embed_lookup


if __name__ == '__main__':
    tf.app.run()
