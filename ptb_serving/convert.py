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
import re

import pandas as pd
import numpy as np
import tensorflow as tf


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


def _build_vocab(filename):
    data = pd.read_csv(filename)

    request_url = data['request_url'].tolist()
    filtered_request_url = [_filter_url(url) for url in request_url]
    referer_url = data[data['referer_url'] != '-1']['referer_url'].tolist()
    filtered_referer_url = [_filter_url(url) for url in referer_url]

    urls = filtered_request_url + filtered_referer_url

    counter = collections.Counter(urls)
    count_pairs = sorted(counter.iteritems(), key=lambda x: (-x[1], x[0]))
    # print(len(count_pairs))
    # ranked_urls, _ = list(zip(*count_pairs))
    # 过滤掉出现次数太少的url
    ranked_urls = [k for k, v in count_pairs if v >= 5]
    print(len(ranked_urls))

    # 词表的大小为url的数目加上2, id=0表示未知词，id=1表示段结束
    url_to_id = dict(zip(ranked_urls, [i+2 for i in range(len(ranked_urls))]))
    url_to_id['UNK'] = 0
    url_to_id['<eos>'] = 1
    id_to_url = dict(zip([i+2 for i in range(len(ranked_urls))], ranked_urls))
    id_to_url[0] = 'UNK'
    id_to_url[1] = '<eos>'
    print("build vocabulary successfully!")

    output = open("data/vocabulary/url_to_id.pkl", 'wb')
    pickle.dump(url_to_id, output)
    output.close()
    output = open("data/vocabulary/id_to_url.pkl", 'wb')
    pickle.dump(id_to_url, output)
    output.close()
    print("Save vocabulary successfully!")

    return url_to_id, id_to_url


def _path_to_ids(path, url_to_id):
    path_ids = list()
    for url in path:
        if url in url_to_id:
            path_ids.append(url_to_id[url])
        else:
            path_ids.append(url_to_id['UNK'])
    return path_ids


def _routes_to_countrynumber(filename, url_to_id, min_path_len=3):
    routes = list()
    routes_to_ipto = list()
    routes_to_ctynumber = list()

    data = pd.read_csv(filename, chunksize=100000, encoding='utf-8')
    print("process data...")
    cur_ipto = 0   # current iptonumber
    path = []
    path_to_ipto = []
    path_to_cty = []
    count = 0
    for chunk in data:
        print("process chunk_{}".format(count))
        sorted_chunk = chunk.sort_values(['iptonumber', 'visit_time'])
        filter_chunk = sorted_chunk[['visit_time', 'iptonumber', 'request_url', 'referer_url', 'country_number']]

        rows_iter = filter_chunk.itertuples()

        # row[0]: index, row[1]: visit_time, row[2]: iptonumber, row[3]: request_url, row[4]: referer_url, row[5]: country_number

        for row in rows_iter:
            if row[2] != cur_ipto:
                if(len(path)>=min_path_len):
                    path_ids = _path_to_ids(path+['<eos>'], url_to_id)
                    path_to_ipto.append(0)   # append country_number for <eos>
                    path_to_cty.append(0)   # append country_number for <eos>
                    map(lambda x: routes.append(x), path_ids)
                    map(lambda x: routes_to_ipto.append(x), path_to_ipto)
                    map(lambda x: routes_to_ctynumber.append(x), path_to_cty)
                path = []
                path_to_ipto = []
                path_to_cty = []
                request_url = _filter_url(row[3])
                referer_url = _filter_url(row[4])
                if(referer_url!='-1'):
                    path.append(referer_url)
                    path_to_ipto.append(row[2])
                    path_to_cty.append(row[5])

                path.append(request_url)
                path_to_ipto.append(row[2])
                path_to_cty.append(row[5])
                if(len(path)>0):
                    cur_ipto = row[2]
            else:
                request_url = _filter_url(row[3])
                referer_url = _filter_url(row[4])
                if (referer_url!='-1' and (referer_url!=path[len(path)-1] or len(path)==0)):
                    path.append(referer_url)
                    path_to_ipto.append(row[2])
                    path_to_cty.append(row[5])
                path.append(request_url)
                path_to_ipto.append(row[2])
                path_to_cty.append(row[5])

        count += 1

    if(len(path)>=min_path_len):
        path_ids = _path_to_ids(path+['<eos>'], url_to_id)
        path_to_ipto.append(0)   # append country_number for <eos>
        path_to_cty.append(0)   # append country_number for <eos>
        map(lambda x: routes.append(x), path_ids)
        map(lambda x: routes_to_ipto.append(x), path_to_ipto)
        map(lambda x: routes_to_ctynumber.append(x), path_to_cty)

    print("process data successfully!")

    output_file = open('data/vocabulary/routes_to_countrynumber.pkl', 'wb')
    pickle.dump(routes_to_ctynumber, output_file)
    output_file.close()
    output_file = open('data/vocabulary/routes_to_iptonumber.pkl', 'wb')
    pickle.dump(routes_to_ipto, output_file)
    output_file.close()


def _routes_contain_particular_url_to_countrynumber(filename, url_to_id, target_url, min_path_len=3):
    routes = list()
    routes_to_ipto = list()
    routes_to_ctynumber = list()

    data = pd.read_csv(filename, chunksize=100000, encoding='utf-8')
    print("process data...")
    cur_ipto = 0   # current iptonumber
    path = []
    path_to_ipto = []
    path_to_cty = []
    count = 0
    target = False
    for chunk in data:
        print("process chunk_{}".format(count))
        sorted_chunk = chunk.sort_values(['iptonumber', 'visit_time'])
        filter_chunk = sorted_chunk[['visit_time', 'iptonumber', 'request_url', 'referer_url', 'country_number']]

        rows_iter = filter_chunk.itertuples()

        # row[0]: index, row[1]: visit_time, row[2]: iptonumber, row[3]: request_url, row[4]: referer_url, row[5]: country_number

        for row in rows_iter:
            if row[2] != cur_ipto:
                if(len(path)>=min_path_len) and target:
                    path_ids = _path_to_ids(path+['<eos>'], url_to_id)
                    path_to_ipto.append(0)   # append country_number for <eos>
                    path_to_cty.append(0)   # append country_number for <eos>
                    map(lambda x: routes.append(x), path_ids)
                    map(lambda x: routes_to_ipto.append(x), path_to_ipto)
                    map(lambda x: routes_to_ctynumber.append(x), path_to_cty)
                path = []
                path_to_ipto = []
                path_to_cty = []
                target = False
                if target_url == row[3] or target_url == row[4]:
                    target = True
                request_url = _filter_url(row[3])
                referer_url = _filter_url(row[4])
                if(referer_url!='-1'):
                    path.append(referer_url)
                    path_to_ipto.append(row[2])
                    path_to_cty.append(row[5])

                path.append(request_url)
                path_to_ipto.append(row[2])
                path_to_cty.append(row[5])
                if(len(path)>0):
                    cur_ipto = row[2]
            else:
                if target_url == row[3] or target_url == row[4]:
                    target = True
                request_url = _filter_url(row[3])
                referer_url = _filter_url(row[4])
                if (referer_url!='-1' and (referer_url!=path[len(path)-1] or len(path)==0)):
                    path.append(referer_url)
                    path_to_ipto.append(row[2])
                    path_to_cty.append(row[5])
                path.append(request_url)
                path_to_ipto.append(row[2])
                path_to_cty.append(row[5])

        count += 1

    if(len(path)>=min_path_len) and target:
        path_ids = _path_to_ids(path+['<eos>'], url_to_id)
        path_to_ipto.append(0)   # append country_number for <eos>
        path_to_cty.append(0)   # append country_number for <eos>
        map(lambda x: routes.append(x), path_ids)
        map(lambda x: routes_to_ipto.append(x), path_to_ipto)
        map(lambda x: routes_to_ctynumber.append(x), path_to_cty)

    print("process data successfully!")

    output_file = open('data/vocabulary/routes_to_countrynumber_contain_particular_url.pkl', 'wb')
    pickle.dump(routes_to_ctynumber, output_file)
    output_file.close()
    output_file = open('data/vocabulary/routes_to_iptonumber_contain_particular_url.pkl', 'wb')
    pickle.dump(routes_to_ipto, output_file)
    output_file.close()


def _cell_state_to_countrynumber(batch_size, num_steps):

    cell_state_to_ctynumber = list()
    input_file = open('data/vocabulary/routes_to_countrynumber.pkl', 'rb')
    routes_to_ctynumber = pickle.load(input_file)
    input_file.close()

    data_len = len(routes_to_ctynumber)
    batch_len = data_len // batch_size

    country_numbers = routes_to_ctynumber[0: batch_size*batch_len]
    epoch_size = (batch_len - 1) // num_steps

    for i in range(epoch_size):
        batch_ctynumber = np.array(country_numbers[i*(batch_size*num_steps):(i+1)*(batch_size*num_steps)])
        batch_ctynumber = np.reshape(batch_ctynumber, [batch_size, num_steps])
        print(batch_ctynumber.shape)
        for j in range(batch_size):
            cell_state_to_ctynumber.append(batch_ctynumber[j, 0])

    output_file = open('data/vocabulary/cell_state_to_countrynumber.pkl', 'wb')
    pickle.dump(cell_state_to_ctynumber, output_file)
    output_file.close()


def _cell_state_to_countrynumber_contain_particular_url(batch_size, num_steps):

    cell_state_to_ctynumber = list()
    input_file = open('data/vocabulary/routes_to_countrynumber_contain_particular_url.pkl', 'rb')
    routes_to_ctynumber = pickle.load(input_file)
    input_file.close()

    data_len = len(routes_to_ctynumber)
    batch_len = data_len // batch_size
    country_numbers = routes_to_ctynumber[0: batch_size*batch_len]
    epoch_size = (batch_len - 1) // num_steps

    for i in range(epoch_size):
        batch_ctynumber = np.array(country_numbers[i*(batch_size*num_steps):(i+1)*(batch_size*num_steps)])
        batch_ctynumber = np.reshape(batch_ctynumber, [batch_size, num_steps])
        for j in range(batch_size):
            cell_state_to_ctynumber.append(batch_ctynumber[j, 0])

    output_file = open('data/vocabulary/cell_state_to_countrynumber_contain_particular_url.pkl', 'wb')
    pickle.dump(cell_state_to_ctynumber, output_file)
    output_file.close()


def main(_):
    # for test
    batch_size = 25
    num_steps = 10
    data_path = "/home/zyt/data/JD/"
    train_path = os.path.join(data_path, "20160720.csv")
    target_url = "bigtree.en.made-in-china.com/"
    url_to_id, _ = _build_vocab(train_path)
    # produce routes_to_ctynumber, routes_to_iptonumber
    _routes_to_countrynumber(train_path, url_to_id)
    # produce cell_state_to_ctynumber
    _cell_state_to_countrynumber(batch_size, num_steps)
    # produce routes_to_ctynumber_contain_particular_url, routes_to_iptonumber_contain_particular_url
    _routes_contain_particular_url_to_countrynumber(train_path, url_to_id, target_url)
    # produce cell_state_to_ctynumber_contain_particular_url
    _cell_state_to_countrynumber_contain_particular_url(batch_size, num_steps)

if __name__ == "__main__":
    tf.app.run()




