#!/bin/bash

RES_FILE=$1

REF_FILE=reference_videos_Youtube.json
echo "evaluating on ${REF_FILE}"
python eval.py ${REF_FILE} ${RES_FILE}

:<<!
REF_FILE=reference_videos_Youtube_len4.json
echo "evaluating on ${REF_FILE}"
python eval.py ${REF_FILE} ${RES_FILE}

REF_FILE=reference_videos_Youtube_len5.json
echo "evaluating on ${REF_FILE}"
python eval.py ${REF_FILE} ${RES_FILE}

REF_FILE=reference_videos_Youtube_len6.json
echo "evaluating on ${REF_FILE}"
python eval.py ${REF_FILE} ${RES_FILE}
!

REF_FILE=reference_videos_Youtube_len7.json
echo "evaluating on ${REF_FILE}"
python eval.py ${REF_FILE} ${RES_FILE}

REF_FILE=reference_videos_Youtube_len8.json
echo "evaluating on ${REF_FILE}"
python eval.py ${REF_FILE} ${RES_FILE}

