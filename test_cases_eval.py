from eval import eval_ceaf

if __name__ == "__main__":

    print("================= case 1 (in the paper) =================")
    golds = {
             "docid1": {"Target": [["Pilmai telephone company building", "telephone company building", "telephone company offices"], ["water pipes"], ["public telephone booth"]]},
            }
    preds = {
             "docid1": {"Target": [["water pipes"], ["Pilmai telephone company building"], ["public telephone booth"], ["telephone company offices"]]},
            }

    results = eval_ceaf(preds, golds, docids=[])
    print("golds", golds)
    print("preds", preds)
    print("phi_strict: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["strict"]["micro_avg"]["p"] * 100, results["strict"]["micro_avg"]["r"] * 100, results["strict"]["micro_avg"]["f1"] * 100))
    print("phi_prop: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["prop"]["micro_avg"]["p"] * 100, results["prop"]["micro_avg"]["r"] * 100, results["prop"]["micro_avg"]["f1"] * 100))
    print

    print("================= case 2 (in the paper) =================")
    golds = {
             "docid1": {"Target": [["Pilmai telephone company building", "telephone company building", "telephone company offices"], ["water pipes"], ["public telephone booth"]]},
            }
    preds = {
             "docid1": {"Target": [["Pilmai telephone company building"], ["water pipes"], ["public telephone booth"]]},
            }

    results = eval_ceaf(preds, golds, docids=[])
    print("golds", golds)
    print("preds", preds)
    print("phi_strict: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["strict"]["micro_avg"]["p"] * 100, results["strict"]["micro_avg"]["r"] * 100, results["strict"]["micro_avg"]["f1"] * 100))
    print("phi_prop: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["prop"]["micro_avg"]["p"] * 100, results["prop"]["micro_avg"]["r"] * 100, results["prop"]["micro_avg"]["f1"] * 100))
    print

    print("================= case 3 (in the paper) =================")
    golds = {
             "docid1": {"Target": [["Pilmai telephone company building", "telephone company building", "telephone company offices"], ["water pipes"], ["public telephone booth"]]},
            }
    preds = {
             "docid1": {"Target": [["Pilmai telephone company building"], ["public telephone booth"]]},
            }

    results = eval_ceaf(preds, golds, docids=[])
    print("golds", golds)
    print("preds", preds)
    print("phi_strict: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["strict"]["micro_avg"]["p"] * 100, results["strict"]["micro_avg"]["r"] * 100, results["strict"]["micro_avg"]["f1"] * 100))
    print("phi_prop: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["prop"]["micro_avg"]["p"] * 100, results["prop"]["micro_avg"]["r"] * 100, results["prop"]["micro_avg"]["f1"] * 100))
    print

    print("\n\n================= case 4 =================")
    golds = {
             "docid1": {"PerpInd": [["m1", "m2", "m3"], ["m4"], ["m5"], ["m6"]]},
            }
    preds = {
             "docid1": {"PerpInd": [["m4"], ["m1"], ["m5"], ["m6"], ["m3"]]},
            }
    results = eval_ceaf(preds, golds, docids=[])
    print("golds", golds)
    print("preds", preds)
    print("phi_strict: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["strict"]["micro_avg"]["p"] * 100, results["strict"]["micro_avg"]["r"] * 100, results["strict"]["micro_avg"]["f1"] * 100))
    print("phi_prop: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["prop"]["micro_avg"]["p"] * 100, results["prop"]["micro_avg"]["r"] * 100, results["prop"]["micro_avg"]["f1"] * 100))
    print


    print("================= case 5 =================")
    golds = {
             "docid2": {"PerpInd": [["m1", "m2", "m3"], ["m4"], ["m5"], ["m6"]]},
            }
    preds = {
             "docid2": {"PerpInd": [["m1", "m2"], ["m4"], ["m5"], ["m6"]]},
            }
    results = eval_ceaf(preds, golds, docids=[])

    print("golds", golds)
    print("preds", preds)
    print("phi_strict: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["strict"]["micro_avg"]["p"] * 100, results["strict"]["micro_avg"]["r"] * 100, results["strict"]["micro_avg"]["f1"] * 100))
    print("phi_prop: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["prop"]["micro_avg"]["p"] * 100, results["prop"]["micro_avg"]["r"] * 100, results["prop"]["micro_avg"]["f1"] * 100))
    print


    print("================= case 6 =================")
    golds = {
             "docid3": {"PerpInd": [["m1", "m2", "m3"], ["m4"], ["m5"], ["m6"]]},
            }
    preds = {
             "docid3": {"PerpInd": [["m1", "m2", "m3", "m4"], ["m5"], ["m6"]]},
            }
    results = eval_ceaf(preds, golds, docids=[])

    print("golds", golds)
    print("preds", preds)
    
    print("phi_strict: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["strict"]["micro_avg"]["p"] * 100, results["strict"]["micro_avg"]["r"] * 100, results["strict"]["micro_avg"]["f1"] * 100))
    print("phi_prop: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["prop"]["micro_avg"]["p"] * 100, results["prop"]["micro_avg"]["r"] * 100, results["prop"]["micro_avg"]["f1"] * 100))
    print


    print("================= case 7 =================")
    golds = {
             "docid1": {"PerpInd": [["m1", "m2", "m3"], ["m4"], ["m5"], ["m6"]]},
             "docid2": {"PerpInd": [["m1", "m2", "m3"], ["m4"], ["m5"], ["m6"]]},
             "docid3": {"PerpInd": [["m1", "m2", "m3"], ["m4"], ["m5"], ["m6"]]},
             "docid4": {"PerpInd": [["m1", "m2", "m3"], ["m4"], ["m5"], ["m6"]],
                        "PerpOrg": [["m1", "m2", "m3"], ["m4"], ["m5"], ["m6"]],
                        "Target": [["m1", "m2", "m3"], ["m4"], ["m5"], ["m6"]]}
            }
    preds = {
             "docid1": {"PerpInd": [["m4"], ["m1"], ["m5"], ["m6"], ["m3"]]},
             "docid2": {"PerpInd": [["m1", "m2"], ["m4"], ["m5"], ["m6"]]},
             "docid3": {"PerpInd": [["m1", "m2", "m3", "m4"], ["m5"], ["m6"]]},
             "docid4": {"PerpInd": [["m4"], ["m1"], ["m5"], ["m6"], ["m3"]],
                        "PerpOrg": [["m1", "m2"], ["m4"], ["m5"], ["m6"]],
                        "Target": [["m1", "m2", "m3", "m4"], ["m5"], ["m6"]]}
            }
    results = eval_ceaf(preds, golds, docids=[])

    print("golds", golds)
    print("preds", preds)
    print("phi_strict: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["strict"]["micro_avg"]["p"] * 100, results["strict"]["micro_avg"]["r"] * 100, results["strict"]["micro_avg"]["f1"] * 100))
    print("phi_prop: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["prop"]["micro_avg"]["p"] * 100, results["prop"]["micro_avg"]["r"] * 100, results["prop"]["micro_avg"]["f1"] * 100))
    print





