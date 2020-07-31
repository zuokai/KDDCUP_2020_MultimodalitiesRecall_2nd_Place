# -*- coding: utf-8 -*-
import os
import io
import csv
import operator
#os.system("python2 imagebert_zk/evaluate_normal.py")
#os.system("python2 imagebert_zk/evaluate_normal_sen2fs.py")
os.system("python2 imagebert_lds/src/run_pretraining_predict_score.py")
#os.system("PYTHONPATH=$PYTHONPATH:./lxmert/src python2 lxmert/src/tasks/kdd.py")

dict_eval = {}
for line in open("../prediction_result/testB_result_match_keyword_valid_finetune_251.txt"):
    arr = line.strip().split("\t")
    if arr[0] not in dict_eval:
        dict_eval[arr[0]] = {}
    dict_eval[arr[0]][arr[1]] = float(arr[2])

dict_eval2 = {}
for line in open("../prediction_result/testB_result_match_keyword_valid_finetune_251_sen_to_forest.txt"):
    arr = line.strip().split("\t")
    if arr[0] not in dict_eval2:
        dict_eval2[arr[0]] = {}
    dict_eval2[arr[0]][arr[1]] = float(arr[2])

dict_eval3 = {}
for line in open("../prediction_result/testBscore_imagebert.txt"):
    arr = line.strip().split("\t")
    if arr[0] not in dict_eval3:
        dict_eval3[arr[0]] = {}
    dict_eval3[arr[0]][arr[1]] = float(arr[2])

dict_eval4 = {}
for line in open("../prediction_result/testB_score_lxmert.csv"):
    arr = line.strip().split(",")
    if "query" in line:
        continue
    if arr[0] not in dict_eval4:
        dict_eval4[arr[0]] = {}
    dict_eval4[arr[0]][arr[1]] = float(arr[2])

dict_product_id_score = {}
dict_product_id_scores = {}
dict_eval_merge = {}
for query_id in dict_eval:
    result = dict_eval[query_id]
    result2 = dict_eval2[query_id]
    result3 = dict_eval3[query_id]
    result4 = dict_eval4[query_id]
    for product_id in result4:
        if product_id not in result:
            result[product_id] = result4[product_id]
            print("x")
        if product_id not in result2:
            result2[product_id] = result4[product_id]
            print("y")
        if product_id not in result3:
            result3[product_id] = result4[product_id]
            print("z")
        merge_score = 0.2*result[product_id] + 0.2*result2[product_id] + 0.3 * result3[product_id] + 0.3 * result4[product_id]
        if query_id not in dict_eval_merge:
            dict_eval_merge[query_id] = {}
            dict_eval_merge[query_id][product_id] = merge_score
        else:
            dict_eval_merge[query_id][product_id] = merge_score
        if product_id not in dict_product_id_score:
            dict_product_id_score[product_id] = merge_score
        elif merge_score > dict_product_id_score[product_id]:
            dict_product_id_score[product_id] = merge_score
        if product_id not in dict_product_id_scores:
            dict_product_id_scores[product_id] = [merge_score]
        else:
            dict_product_id_scores[product_id].append(merge_score)

dict_eval_merge_top1 = {}

for query_id in dict_eval_merge:
    for product_id in dict_eval_merge[query_id]:
        a = dict_product_id_scores[product_id]
        a.sort(reverse=True)
        if len(dict_product_id_scores[product_id]) >= 2:
            if a[0] - a[1] < 0.92:
                continue
        if abs(dict_eval_merge[query_id][product_id] - dict_product_id_score[product_id]) < 1e-5:
            if query_id not in dict_eval_merge_top1:
                dict_eval_merge_top1[query_id] = {}
            dict_eval_merge_top1[query_id][product_id] = dict_eval_merge[query_id][product_id]

f = io.open('../prediction_result/submission.csv','wb')
csv_writer = csv.writer(f)
csv_writer.writerow([u"query-id",u"product1",u"product2",u"product3",u"product4",u"product5"])
less_than_5_querys = []
for query_id in dict_eval_merge_top1:
    product_score = dict_eval_merge_top1[query_id]
    sorted_x = sorted(product_score.items(), key=operator.itemgetter(1), reverse=True)
    if len(sorted_x) < 5:
        print("x: ", query_id, sorted_x)
        less_than_5_querys.append(query_id)
        continue
    csv_writer.writerow([query_id, sorted_x[0][0], sorted_x[1][0], sorted_x[2][0], sorted_x[3][0], sorted_x[4][0]])

for query_id in less_than_5_querys:
    product_score = dict_eval_merge[query_id]
    sorted_x = sorted(product_score.items(), key=operator.itemgetter(1), reverse=True)
    csv_writer.writerow([query_id, sorted_x[0][0], sorted_x[1][0], sorted_x[2][0], sorted_x[3][0], sorted_x[4][0]])