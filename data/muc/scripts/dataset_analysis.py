import json
import os
from collections import OrderedDict
from collections import Counter
from transformers import BertTokenizer
tag2role = OrderedDict({'perp_individual_id': "PerpInd", 'perp_organization_id': "PerpOrg", 'phys_tgt_id': "Target", 'hum_tgt_name': "Victim", 'incident_instrument_id': "Weapon"})

if __name__=='__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # doc lengths distribution
    dataset_all_lengths = []
    dataset_all_para_num = []

    for div in ["train", "dev", "test"]:
        doc_file = "../raw_files/proc_output/" + "doc_" + div
        doc_dict = dict()

        lengths = []
        with open(doc_file, "r") as f_doc:
            for line in f_doc:
                line_json = json.loads(line)
                doc_text = " ".join(line_json["text"].split("\n\n")).lower()
                doc_dict[line_json["docid"]] = doc_text

                doc_text_tokens = tokenizer.tokenize(doc_text)
                length = len(doc_text_tokens)
                lengths.append(length)

                dataset_all_lengths.append(len(doc_text_tokens))
                dataset_all_para_num.append(len(line_json["text"].split("\n\n")))

                # if length > 512:
                #     print(line_json["docid"])

        lengths_1 = [] # < 128
        lengths_2 = [] # 128 <= x < 256
        lengths_3 = [] # 256 <= x < 512
        lengths_4 = [] # 512 <= x
        for l in lengths:
            if l < 128:
                lengths_1.append(l)
            elif l < 256:
                lengths_2.append(l)
            elif l < 480:
                lengths_3.append(l)
            else:
                lengths_4.append(l)

        str_print = "===== {} =====\n".format(div)
        str_print += "length range,  # docs, prop \n"
        str_print += "{}, {}, {:.2f} \n".format("l < 128       ", len(lengths_1), len(lengths_1)/len(lengths))
        str_print += "{}, {}, {:.2f} \n".format("128 <= l < 256", len(lengths_2), len(lengths_2)/len(lengths))
        str_print += "{}, {}, {:.2f} \n".format("256 <= l < 512", len(lengths_3), len(lengths_3)/len(lengths))
        str_print += "{}, {}, {:.2f} \n".format("480 <= l      ", len(lengths_4), len(lengths_4)/len(lengths))

        print(str_print)

    
    print("avg length of doc:", sum(dataset_all_lengths)/len(dataset_all_lengths))
    print("avg # of paragraphs per doc:", sum(dataset_all_para_num)/len(dataset_all_para_num))
    print()
    # import ipdb; ipdb.set_trace()

    # max num of entitys per article (20)
    doc_num_of_entitys = []
    event_cnt = 0
    for div in ["train", "dev", "test"]:
        data_dir = "../processed/"
        file_path = os.path.join(data_dir, "{}.json".format(div))
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                extracts = line["extracts"]
                num_of_entitys = 0
                for _, role in tag2role.items():
                    num_of_entitys += len(extracts[role])
                doc_num_of_entitys.append(num_of_entitys)
                if num_of_entitys > 15:
                    print(div, "example num_of_entitys > 15")
                if num_of_entitys != 0:
                    event_cnt += 1

    print("max_num_of_entitys (per doc): ", max(doc_num_of_entitys))
    print("avg_num_of_entitys (per doc): ", sum(doc_num_of_entitys)/len(doc_num_of_entitys))
    print("distribution: ", Counter(doc_num_of_entitys).most_common) # 0: 794; total: 1700

    print("terrorism event cnt: ", event_cnt)


    # # maximum length of mention
    # mention_lengths = []
    # for div in ["train", "dev", "test"]:
    #     data_dir = "../processed/"
    #     file_path = os.path.join(data_dir, "{}.json".format(div))
    #     with open(file_path, encoding="utf-8") as f:
    #         for line in f:
    #             line = json.loads(line)
    #             extracts = line["extracts"]
    #             for _, role in tag2role.items():
    #                 for entity in extracts[role]:
    #                     for m in entity:
    #                         mention_lengths.append(len(tokenizer.tokenize(m[0])))
    # print("mention lengths distribution: ", Counter(mention_lengths).most_common) # len < 30
    # import ipdb; ipdb.set_trace()
    # "freddy , and several of his bodyguards were killed , president virgilio barco today called an urgent council of ministers meeting to discuss the situation and receive a full report from military and police authorities . here is the second part of the news conference featuring das director general miguel maza marquez"

