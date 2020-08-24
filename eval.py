import numpy as np
import re
import string
import json
import argparse
from scipy.optimize import linear_sum_assignment # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
from collections import OrderedDict
tag2role = OrderedDict({'perp_individual_id': "PerpInd", 'perp_organization_id': "PerpOrg", 'phys_tgt_id': "Target", 'hum_tgt_name': "Victim", 'incident_instrument_id': "Weapon"})



def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


def phi_strict(c1, c2):
    # similarity: if c2 (pred) is subset of c1 (gold) return 1 
    for m in c2:
        if m not in c1:
            return 0
    return 1


def phi_prop(c1, c2):
    # # similarity: len(overlap of c2 (pred) and c1 (gold)) / len(c2)
    return len([m for m in c1 if m in c2]) / len(c2)

def ceaf(clusters, gold_clusters, phi_similarity):
    # !!! need to comment the next line, the conll-2012 eval ignore singletons
    # clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi_similarity(gold_clusters[i], clusters[j])
    # matching = linear_assignment(-scores) # [deprecated] linear_assignment from sklearn
    # similarity = sum(scores[matching[:, 0], matching[:, 1]])
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = sum(scores[row_ind, col_ind])
    return similarity, len(clusters), similarity, len(gold_clusters)


def eval_ceaf_base(preds, golds, phi_similarity, docids=[]):
    result = OrderedDict()
    all_keys = list(role for _, role in tag2role.items()) + ["micro_avg"]
    for key in all_keys:
        result[key] = {"p_num": 0, "p_den": 0, "r_num": 0, "r_den": 0, "p": 0, "r": 0, "f1": 0}

    if not docids:
        for docid in golds:
            docids.append(docid)

    for docid in docids:
        pred = preds[docid]
        gold = golds[docid]

        for role in gold:
            pred_clusters = []
            gold_clusters = []
            for entity in gold[role]:
                gold_c = []
                for mention in entity:
                    gold_c.append(mention)
                gold_clusters.append(gold_c)

            for entity in pred[role]:
                pred_c = []
                for mention in entity:
                    pred_c.append(mention)
                pred_clusters.append(pred_c)

            pn, pd, rn, rd = ceaf(pred_clusters, gold_clusters, phi_similarity)
            result[role]["p_num"] += pn
            result[role]["p_den"] += pd
            result[role]["r_num"] += rn
            result[role]["r_den"] += rd

    result["micro_avg"]["p_num"] = sum(result[role]["p_num"] for _, role in tag2role.items())
    result["micro_avg"]["p_den"] = sum(result[role]["p_den"] for _, role in tag2role.items())
    result["micro_avg"]["r_num"] = sum(result[role]["r_num"] for _, role in tag2role.items())
    result["micro_avg"]["r_den"] = sum(result[role]["r_den"] for _, role in tag2role.items())
    
    
    for key in all_keys:
        result[key]["p"] = 0 if result[key]["p_num"] == 0 else result[key]["p_num"] / float(result[key]["p_den"])
        result[key]["r"] = 0 if result[key]["r_num"] == 0 else result[key]["r_num"] / float(result[key]["r_den"])
        result[key]["f1"] = f1(result[key]["p_num"], result[key]["p_den"], result[key]["r_num"], result[key]["r_den"])

    return result


def normalize_string(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def eval_ceaf(preds, golds, docids=[]):
    # normalization mention strings
    for docid in preds:
        for role in preds[docid]:
            for idx in range(len(preds[docid][role])):
                for idy in range(len(preds[docid][role][idx])):
                    preds[docid][role][idx][idy] = normalize_string(preds[docid][role][idx][idy])
    for docid in golds:
        for role in golds[docid]:
            for idx in range(len(golds[docid][role])):
                for idy in range(len(golds[docid][role][idx])):
                    golds[docid][role][idx][idy] = normalize_string(golds[docid][role][idx][idy])

    results_strict = eval_ceaf_base(preds, golds, phi_strict, docids)
    results_prop = eval_ceaf_base(preds, golds, phi_prop, docids)

    final_results = OrderedDict()
    final_results["strict"] = results_strict
    final_results["prop"] = results_prop

    return final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", default=None, type=str, required=False, help="preds output file")
    parser.add_argument("--gold_file", default="./data/muc/processed/test.json", type=str, required=False, help="gold file")
    args = parser.parse_args()

    ## get pred and gold extracts
    preds = OrderedDict()
    golds = OrderedDict()
    with open(args.pred_file, encoding="utf-8") as f:
        out_dict = json.load(f)
        for docid in out_dict:
            preds[docid] = out_dict[docid]["pred_extracts"]

    with open(args.gold_file, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            docid = str(int(line["docid"].split("-")[0][-1])*10000 + int(line["docid"].split("-")[-1]))

            extracts_raw = line["extracts"]

            extracts = OrderedDict()
            for role, entitys_raw in extracts_raw.items():
                extracts[role] = []
                for entity_raw in entitys_raw:
                    entity = []
                    for mention_offset_pair in entity_raw:
                        entity.append(mention_offset_pair[0])
                    if entity:
                        extracts[role].append(entity)
            golds[docid] = extracts

    # import ipdb; ipdb.set_trace()
    docids = []
    results = eval_ceaf(preds, golds, docids)
    all_keys = list(role for _, role in tag2role.items()) + ["micro_avg"]
    str_print = []
    for key in all_keys:
        if key == "micro_avg":
            print("***************** {} *****************".format(key))
        else:
            print("================= {} =================".format(key))
            
        str_print += [results["strict"][key]["p"] * 100, results["strict"][key]["r"] * 100, results["strict"][key]["f1"] * 100]
        print("P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["strict"][key]["p"] * 100, results["strict"][key]["r"] * 100, results["strict"][key]["f1"] * 100)) # phi_strict
        # print("phi_prop: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["prop"][key]["p"] * 100, results["prop"][key]["r"] * 100, results["prop"][key]["f1"] * 100))
        print()
    str_print= ["{:.2f}".format(r) for r in str_print]
    print("print: {}".format(" ".join(str_print)))
        



