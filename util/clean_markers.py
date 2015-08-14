#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on Nov 1, 2014

@author: rmn
'''
import re
from nltk.tokenize.regexp import RegexpTokenizer
tokenizer = RegexpTokenizer('[^\w\-\']+', gaps=True)
reg_apa = re.compile(
    # [Chen et al.2000]
    r"\(\s?(([A-Za-z\-]+\s)+([A-Za-z\-]+\.?)?,?\s\d{2,4}(;\s)?)+\s?\)|"
    r"\(\s?([^ ]+\s?[^ ]*et\sal\.,?\s\d{2,4}(,\s)?)+(\sand\s)?[^ ]+\s?[^ ]*et\sal\.,?\s\d{2,4}\)|"
    r"\w+\set al\. \(\d{2,4}\)")  # [Chen et al. 200]
reg_apa_rare = re.compile(
    r"((([A-Z]\w+\set\sal\.,? \d{4})|([A-Z]\w+\sand\s[A-Z]\w+,?\s\d{4}))((,\s)| and )?)+")
reg_apa2 = re.compile(
    r"\(\s?(\w+\s?\w*et\sal\.,\s\d{2,4}(,\s)?)+(\sand\s)?\w+\s?\w*et\sal\.,\s\d{2,4}\)")
reg_ieee = re.compile(r"(\[(\d+([,–]\s?)?)+\])|\[\s?[\d,-]+\]")
reg_paranthesis = re.compile(
    r"\(\s?\d{1,2}(,\s\d{1,2})*(\sand\s\d{1,2})?\)")
with file('data/stopwords.txt') as f:
    stopwords = frozenset([l.strip().lower() for l in f])


def clean(txt, remove_stopwords=False):
    cleaned = reg_apa.sub('', txt)
    cleaned = reg_ieee.sub('', cleaned)
    cleaned = reg_paranthesis.sub('', cleaned)
    cleaned = reg_apa_rare.sub('', cleaned)
    cleaned = re.sub('\s+', ' ', cleaned).strip()
    cleaned = re.sub('(,\s)+', ', ', cleaned).strip(', ')
    if remove_stopwords:
        cleaned = ' '.join([w for w in tokenizer.tokenize(cleaned)
                            if w not in stopwords])
    return cleaned
