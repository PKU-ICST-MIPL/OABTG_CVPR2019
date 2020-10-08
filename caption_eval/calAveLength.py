import json as js
import sys

if __name__ == '__main__':
  res_path = sys.argv[1]
  total_len = 0
  with open(res_path,'r') as res_file:
    results = js.load(res_file)
  results = results['val_predictions']
  res_cnt = len(results)
  for res_i in range(res_cnt):
    cur_res = results[res_i]
    cur_res_sent_len = len(cur_res['caption'].strip().split(' '))
    total_len+=cur_res_sent_len
  ave_len = total_len * 1.0 / res_cnt
  print('average length = %f' % ave_len)