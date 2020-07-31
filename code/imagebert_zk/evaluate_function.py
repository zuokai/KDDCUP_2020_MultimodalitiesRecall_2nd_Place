import numpy as np
from tqdm import tqdm
import json

def evaluate(match_pred, match_label, rank_score_pred, cal_acc_ndcg="ndcg", k=5):
    """
    cal_acc_ndcg
    """ 
    if cal_acc_ndcg=='acc':     
        return accuracy_score(list(match_pred.values()), list(match_label.values()))
    elif cal_acc_ndcg=='ndcg':
        num_query = len(rank_score_pred)
        rank_label = json.load(open("../inputs/valid_answer.json"))
        ndcg_sum = 0.

        for query_id in rank_label.keys():
            if query_id not in rank_score_pred:
                #print(query_id, rank_score_pred)
                continue
            rlist = rank_score_pred[query_id]
            rlist.sort(key=lambda x: x[1], reverse=True)

            ground_truth_ids = set([str(product_id) for product_id in rank_label[query_id]])
            ref_vec = [1.0] * len(ground_truth_ids)
            pred_vec = [1.0 if product_id_score[0] in ground_truth_ids else 0.0 for product_id_score in rlist]
            ndcg_sum += get_ndcg(pred_vec, ref_vec, k)

        ndcg = ndcg_sum / len(rank_label)

        return ndcg

# compute dcg@k for a single sample
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(3, r.size + 2)))
    return 0.

# compute ndcg@k (dcg@k / idcg@k) for a single sample
def get_ndcg(r, ref, k):
    dcg_max = dcg_at_k(ref, k)
    if not dcg_max:
        return 0.
    dcg = dcg_at_k(r, k)
    return dcg / dcg_max
