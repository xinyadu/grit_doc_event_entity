import json
import os
from collections import OrderedDict
from collections import Counter
from transformers import BertTokenizer
tag2role = OrderedDict({'perp_individual_id': "PerpInd", 'perp_organization_id': "PerpOrg", 'phys_tgt_id': "Target", 'hum_tgt_name': "Victim", 'incident_instrument_id': "Weapon"})


def overlap_e1_e2(e1, e2):
    e1_ts = e1.split()
    e2_ts = e2.split()

    for t1 in e1_ts:
        for t2 in e2_ts:
            if t1 == t2:
                return True
    else:
        return False

if __name__=='__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    div = "test"
    data_dir = "../processed/"
    file_path = os.path.join(data_dir, "{}.json".format(div))
    docids = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            docid = int(line["docid"].split("-")[0][-1])*10000 + int(line["docid"].split("-")[-1])
            extracts = line["extracts"]

            ind_entitys_ms = []            
            for e in extracts["PerpInd"]:
                for m in e:
                    ind_entitys_ms.append(m[0])

            org_entitys_ms = []            
            for e in extracts["PerpOrg"]:
                for m in e:
                    org_entitys_ms.append(m[0])

            # ind_entitys_ms = [m[0] for m in e for e in extracts["PerpInd"]]
            # org_entitys_ms = [m[0] for m in e for e in extracts["PerpOrg"]]


            no_overlap = True
            for e1 in ind_entitys_ms:
                for e2 in org_entitys_ms:
                    # if e1 in e2 or e2 in e1:
                    # if e2 in e1:
                    if overlap_e1_e2(e1, e2):
                        print(e1, e2)
                        no_overlap = False

            if not no_overlap:
                docids.append(str(docid))

    # avg_entity_num_mention_docid = sorted([(docid, avg_num) for docid, avg_num in docid_avg_entity_num_mention.items()], key = lambda pair: pair[1]) 
    with open("../processed/docids_overlap_ind_org.json", "w+") as f:
        f.write(json.dumps(docids, indent=4))
