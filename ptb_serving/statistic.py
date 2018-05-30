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


def _url_count(filename):
    url_count = {}
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
    # ranked_urls = [k for k, v in count_pairs if v >= 5]
    for k, v in count_pairs:
        url_count[k] = v

    output = open("data/statistic/url_count.pkl", 'wb')
    pickle.dump(url_count, output)
    output.close()
    print("save url_count successfully!")


def _route_count(filename):
    route_count = {}

    data = pd.read_csv(filename, chunksize=100000, encoding='utf-8')
    print("process data...")
    cur_ipto = 0   # current iptonumber
    path = []
    count = 0
    for chunk in data:
        print("process chunk_{}".format(count))
        #sorted_chunk = chunk.sort_values(['iptonumber', 'visit_time'])
        filter_chunk = chunk[['visit_time', 'iptonumber', 'request_url', 'referer_url']]

        rows_iter = filter_chunk.itertuples()

        # row[0]: index, row[1]: visit_time, row[2]: iptonumber, row[3]: request_url, row[4]: referer_url

        for row in rows_iter:
            if row[2] != cur_ipto:
                if len(path) in route_count.keys():
                    route_count[len(path)] += 1
                else:
                    route_count[len(path)] = 1

                path = []
                request_url = _filter_url(row[3])
                referer_url = _filter_url(row[4])
                if(referer_url!='-1'):
                    path.append(referer_url)

                path.append(request_url)
                if(len(path)>0):
                    cur_ipto = row[2]
            else:
                request_url = _filter_url(row[3])
                referer_url = _filter_url(row[4])
                if (referer_url!='-1' and (referer_url!=path[len(path)-1] or len(path)==0)):
                    path.append(referer_url)
                path.append(request_url)
            if(len(path)>400):
                print(cur_ipto, row[0])

        count += 1
    print(route_count)
    output = open("data/statistic/route_count.pkl", 'wb')
    pickle.dump(route_count, output)
    output.close()
    print("save route_count successfully!")


def main(_):
    # for test
    data_path = "/home/zyt/data/JD/"
    train_path = os.path.join(data_path, "20160727.csv")

    #_url_count(train_path)
    _route_count(train_path)


if __name__ == "__main__":
    tf.app.run()




