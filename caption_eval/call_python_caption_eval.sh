#!/bin/bash

cd caption_eval
#python /home/zhangjunchao/sam-tensorflow-master/sam-tensorflow-master/caption_eval/eval.py /home/zhangjunchao/sam-tensorflow-master/sam-tensorflow-master/caption_eval/reference_videos_Youtube.json $1
python eval.py reference_videos_Youtube.json $1

cd ../