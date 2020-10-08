#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer

import json
import sys
import os

LANGUAGE = "english"
SENTENCES_COUNT = 1

json_file = sys.argv[1]
js = json.load(open(json_file, 'r'))

outpath = os.path.splitext(json_file)[0]
results = {}  # the result dict cataining the final results

if __name__ == "__main__":
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    for k, candidates in js.iteritems():
        string = ' . '.join(candidates) + ' . '
        parser = PlaintextParser.from_string(string, Tokenizer(LANGUAGE))
        winner = [sent._text for sent in summarizer(parser.document, SENTENCES_COUNT)]
        results[k] = winner

    json.dump(results, open(outpath+'_sumed.json', 'w'))
