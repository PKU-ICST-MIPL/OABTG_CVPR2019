#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider

import json
import sys
import os

# Global variables
GT_ORG_PATH = "/home/yzw/dataset/Youtube/json/youtube_train_val_test_an.json"

class MSR_Evalator:
    """
    The format of input json file:
    1. Reference file:  -- every video has a certain number of captions
        { video_id_0 : [ caption_0, caption_1, ... ],
          video_id_1 : [ caption_0, caption_1, ... ],
            ...
          video_id_n : [ caption_0, caption_1, ... ]
        }
    2. Result file:
        { video_id_0 : [ caption ],  -- every video has only one caption
          video_id_1 : [ caption ],
            ...
          video_id_n : [ caption ],
        }
    """

    def __init__(self, ref, res):
        self.ref = ref  # parsed json file, references
        self.res = res  # parsed json file, results
        # print "################",ref,res

    def evaluate(self):
        # ==================================================
        # Tokenization, remove punctutions
        # =================================================
        '''
        print "tokenization ..."
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(self.ref)
        res = tokenizer.tokenize(self.res)
        '''
        gts = self.ref
        # ==================================================
        # Set up scorers
        # ==================================================
        print "setting up scorers ..."
        scorers = [
            (Bleu(4), ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4")),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # ==================================================
        # Set up scorers
        # ==================================================
        out = {}
        for scorer, method in scorers:
            print "computing %s score ..." %(scorer.method())
            score, scores = scorer.compute_score(gts, res)
            if isinstance(method, tuple):
                for sc, scs, m in zip(score, scores, method):
                    out[m] = sc
                    print "%s: %0.4f" %(m, sc)
            else:
                print "%s: %0.4f" %(method, score)
                out[method] = score

        return out

def rearrange_reference(outpath, jspath):
    js = json.load(open(jspath, 'r'))
    gts = {}
    for x in js['sentences']:
        if not gts.has_key(x['video_id']):
            gts[x['video_id']] = [x['caption']]
        else:
            gts[x['video_id']].append(x['caption'])
    outfp = open(outpath, 'w')
    json.dump(gts, outfp)

def rearrange_ref_youtube(outpath, jspath):
    js = json.load(open(jspath,'r'))
    gts = {}
    for i,img in enumerate(js):
        gts[img['id']] = img['captions']
    outfp = open(outpath,'w')
    json.dump(gts, outfp)
    outfp.close()

def rearrange_results(js):
    lst = js['val_predictions']
    out = {}
    for x in lst:
        out[x['image_id']] = [ x['caption'] ]

    return out

if __name__ == "__main__":
    ref_path = sys.argv[1]; res_path = sys.argv[2]
    #if not os.path.isfile(ref_path): rearrange_reference(ref_path, GT_ORG_PATH)
    if not os.path.isfile(ref_path): rearrange_ref_youtube(ref_path, GT_ORG_PATH)
    ref = json.load(open(ref_path, 'r'))
    res = json.load(open(res_path, 'r'))
    # the result json file generated from train.lua script
    if res.keys()[0] == 'val_predictions':
        res = rearrange_results(res)
    evaluator = MSR_Evalator(ref, res)
    out = evaluator.evaluate()

