# -*- coding: utf-8 -*-

from flask import Flask
from flask import request
from flask import jsonify
import datetime
import hashlib
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import kashgari
import re
import time
import pickle
import os

app = Flask(__name__)
model = None

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = '********@gmail.com'    #
SALT = '*************'                  #
#########################################


def pickle_dump(obj, file_name, output_dir="questions"):
    with open(os.path.join(output_dir, file_name), 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(file_name, input_dir):
    with open(os.path.join(input_dir, file_name), 'rb') as f:
        s = pickle.load(f)
    return s


def split2char(word):
    return [char for char in word]


def split_context_under_max_seqLen(input_content):
    split_new_content_list = []
    for content in input_content:
        if len(content) >= 100:
            sub_split_new_content_list = [content[x:x+99] for x in range(0, len(content), 99)]
            for sub_split_content in sub_split_new_content_list:
                split_new_content_list += [sub_split_content]
        else:
            split_new_content_list += [content]
    return split_new_content_list


def strQ2B(ustring):
    if isinstance(ustring, str) :
      ss = []
      for s in ustring:
          rstring = ""
          for uchar in s:
              inside_code = ord(uchar)
              if inside_code == 12288: 
                  inside_code = 32
              elif (inside_code >= 65281 and inside_code <= 65374):
                  inside_code -= 65248
              rstring += chr(inside_code)
          ss.append(rstring)
      return ''.join(ss)
    else:
      return unstring


def strClean(x):
    if isinstance(x, str):
        x = re.sub("[▲◆▪【】*';％%※★<>#〈〉' '_|｜()（）－-]",  "", strQ2B(x))
        x = re.sub("[!！？?]", "。", x)
    return x


def purge_char(x, char_length=7000):
    return x[:char_length]


def load_model():
    # ref1: https://blog.techbridge.cc/2018/11/01/python-flask-keras-image-predict-api/
    # ref1.1: https://stackoverflow.com/questions/47115946/tensor-is-not-an-element-of-this-graph
    # ref2: https://github.com/tensorflow/tensorflow/issues/28287
    global sess
    global graph
    global model
    global AML_classifier

    # initialize tf graph
    sess = tf.Session()
    graph = tf.get_default_graph()
    # load model
    set_session(sess)
    model = kashgari.utils.load_model('model_all_r_2.h5')
    AML_classifier = kashgari.utils.load_model('model_binary_1.h5')


def data_preprocess(data):
    return split_context_under_max_seqLen([split2char(strClean(data))])


def load_tabu_list():
    global tabu_list

    tabu_list = ["佰萬", "仟萬", "百萬", "千萬", "一億", "兩億", "三億", "四億", "五億", "六億", "七億", "八億", "九億", "十億",
                 "參億", "肆億", "伍億", "陸億", "柒億", "捌億", "玖億", "拾億", "壹億", "貳億",
                 "二億", "億萬", "兆元", "億元", "千元", "萬元", "百元", "一審", "二審", "三審"]


def load_keyword_list():
    global keyword_list

    keyword_list = ['吸金', '收賄', '洗錢','行賄','貪汙', '貪污','貪瀆','回扣', '賄賂', "暴利",'不法獲利', '詐欺', "索賄",
                    '詐欺前科', '詐欺取財', '詐貸', '詐領', '詐騙', '詐取', '榨取', '暴力討債', '販毒', "龐式騙局", "非法獲利", 
                    '證交法', '證券交易法', '地下匯兌', '套利', '匯兌', '內線', "捲款潛逃", "捲款", '人頭戶', '仿冒品', '侵占', 
                    '偽造', '包庇', '弊案', '恐嚇取財', '掏空',  '海洛因', '炒股', '營業祕密法', '白手套', '竊盜', '期貨交易法', 
                    '經濟犯', '老鼠會', '製毒', '買票', '贓款', '走私', '逃漏', '逃漏稅', "逃稅", '銀行法', "挪用", "弊端", 
                    "涉弊", "牟利", "浮報", "虛報", "黑金", "資恐", "恐怖主義", "廉政", "收受", "假帳", "地下錢莊"]


def keyword_classifier(article):
    AML = False
    for keyword in keyword_list:
        if keyword in article:
            AML = True
            break
    return AML


def generate_server_uuid(input_string):
    """ Create your own server_uuid
    @param input_string (str): information to be encoded as server_uuid
    @returns server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string+SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def predict(news):
    """ Predict your model result
    @param article (str): a news article
    @returns prediction (list): a list of name
    """

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######
    # prediction = ['aha','danny','jack']

    raw_text = data_preprocess(purge_char(news, char_length=7000))

    with graph.as_default():
        set_session(sess)

        # if not larger than threshold, return []
        tensor = AML_classifier.embedding.process_x_dataset(raw_text)
        probs = AML_classifier.tf_model.predict(tensor)

        if np.max(probs[:, 1]) < 0.6:
            print("Predict by AML Classifier and returned [].")
            return []
        
        if not keyword_classifier(news):
            print("Predict by Keyword Classifier and returned [].")
            return []
        
        raw_text = data_preprocess(purge_char(news, char_length=3200))
        ners = model.predict(raw_text)

    pred_names = []

    for sect_idx in range(len(raw_text)):
        ner_reg_list = []

        for index, (word, tag) in enumerate(zip(raw_text[sect_idx], ners[sect_idx])):
            if tag != 'O':
                ner_reg_list.append((word, tag, index))

        if ner_reg_list:
            for i, item in enumerate(ner_reg_list):
                if item[1].startswith('B'):
                    label = ""
                    end = i + 1
                    while end <= len(ner_reg_list) - 1 and ner_reg_list[end][1].startswith('I') and (ner_reg_list[end][2] - ner_reg_list[end-1][2]) == 1: 
                        end += 1
                    
                    label += ''.join([item[0] for item in ner_reg_list[i:end]])
                    pred_names.append(label)
    
    # output
    if len(pred_names) > 0:
        pred_names = np.unique([item.replace(' ','') for item in pred_names]).tolist()
        pred_names = [name for name in pred_names if len(name) > 1]
        # pred_names = _check_datatype_to_list(pred_names) # Officially provided

        # tabu list
        for tabu in tabu_list:
            for pred_name in pred_names:
                if tabu in pred_name:
                    pred_names.remove(pred_name)

        # remove duplicated part of names from names
        len_two_name = [name for name in pred_names if len(name) == 2]
        len_three_name = [name for name in pred_names if len(name) > 2]
        
        for name in len_two_name:
            if np.sum([name in name3 for name3 in len_three_name]) > 0:
                pred_names.remove(name)

        return pred_names
    elif len(pred_names) == 0:
        return []
    
    ####################################################
    # prediction = _check_datatype_to_list(prediction)
    # return prediction


def _check_datatype_to_list(prediction):
    """ Check if your prediction is in list type or not. 
        And then convert your prediction to list type or raise error.
        
    @param prediction (list / numpy array / pandas DataFrame): your prediction
    @returns prediction (list): your prediction in list type
    """
    if isinstance(prediction, np.ndarray):
        _check_datatype_to_list(prediction.tolist())
    elif isinstance(prediction, pd.core.frame.DataFrame):
        _check_datatype_to_list(prediction.values)
    elif isinstance(prediction, list):
        return prediction
    raise ValueError('Prediction is not in list type.')


@app.route('/healthcheck', methods=['POST'])
def healthcheck():
    """ API for health check """
    data = request.get_json(force=True)  
    t = datetime.datetime.now()  
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)
    server_timestamp = t.strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'esun_uuid': data['esun_uuid'], 'server_uuid': server_uuid, 'captain_email': CAPTAIN_EMAIL, 'server_timestamp': server_timestamp})


@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API """
    data = request.get_json(force=True)  
    esun_timestamp = data['esun_timestamp']
    
    t = datetime.datetime.now()  
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)
    
    # time 
    start = time.time()

    try:
        # preprocess
        print("Dump before preprocessing...:", data['news'])
        
        t = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        # dump Q
        Q_name = "Q_" + t + ".pickle"
        pickle_dump(obj=data['news'], file_name=Q_name, output_dir="questions")

        answer = predict(data['news'])
        print("answer:", answer)
    except:
        raise ValueError('Model error.')
    
    print("News length:", len(data['news']))
    print("Total predict time:", time.time() - start)

    server_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'esun_timestamp': data['esun_timestamp'], 'server_uuid': server_uuid, 'answer': answer, 'server_timestamp': server_timestamp, 'esun_uuid': data['esun_uuid']})


if __name__ == "__main__":
    graph = tf.get_default_graph()
    load_tabu_list()
    load_keyword_list()
    load_model()
    app.run(host='0.0.0.0', port=80, debug=True) # default: 80, 443
