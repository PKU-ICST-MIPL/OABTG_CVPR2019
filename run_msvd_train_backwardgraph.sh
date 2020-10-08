FLAG="--soft --bidirectional"
PARAMS1="--centers_num=64 --lr=0.0001 --epoch=20 --d_w2v=512 --output_dim=512 --reduction_dim=512"
PARAMS2="--gpu_id=3 --cap_len_max=15 --cap_len_min=3 --input_feature_dim=2048"

INPUT_FEATURE="--input_feature=features/msvd/msvd_all_sample40_frame_resnet200_res5c_relu.h5"

PRE_TRAIN=""

SAVE_MODEL_ROOT="saved_model/msvd/"
SAVE_MODEL_FOLDER="--model_folder=${SAVE_MODEL_ROOT}/youtube_sample40_objSim_NMS0.3_reverse_WV_att_NoShare_resnet200_res5c_relu_capl15s3"
LOG_FILE="${SAVE_MODEL_ROOT}/log_msvd_sam_sample40_objSim_NMS0.3_reverse_WV_att_NoShare_resnet200_res5c_relu_capl15s3"
CMD="python msvd_main_objectV_att_NoShare_reverse.py $FLAG $PARAMS1 $PARAMS2 $INPUT_FEATURE $SAVE_MODEL_FOLDER $PRE_TRAIN"

echo $CMD 
echo $CMD >> $LOG_FILE
$CMD --log_file=${LOG_FILE}
#$CMD 2>&1 | tee $LOG_FILE

wait 
