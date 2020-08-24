"""
process the train/dev/test file
"""

import json
from collections import OrderedDict
# import spacy
tag2role = OrderedDict({'perp_individual_id': "PerpInd", 'perp_organization_id': "PerpOrg", 'phys_tgt_id': "Target", 'hum_tgt_name': "Victim", 'incident_instrument_id': "Weapon"})

def is_subset(candidate, entity):
    for m in candidate:
        if m not in entity:
            return False

    return True

def read_files(div):
    doc_file = "../raw_files/proc_output/" + "doc_" + div
    keys_file = "../raw_files/proc_output/" + "keys_" + div
    doc_dict = OrderedDict()
    keys_dict = OrderedDict()

    # get doc text
    with open(doc_file) as f_doc:
        for line in f_doc:
            line_json = json.loads(line) # odict_keys(['char_before', 'char_end', 'char_start', 'dateline', 'docid', 'source', 'tags', 'text'])
            docid = line_json["docid"]
            doc_text = line_json["text"]
            doc_dict[docid] = doc_text.lower()


    # read keys (extracts) from files and merge the entity from different templates
    with open(keys_file) as f_keys:
        contents = f_keys.read()
        contents = contents.split("%%%")
        for content in contents:
            if not content: continue
            content = json.loads(content)
            key, docid = content[0][0], content[0][1]
            if key == "message_id":
                if docid not in keys_dict:
                    keys_dict[docid] = OrderedDict()
                    for tag, role in tag2role.items():
                        keys_dict[docid][role] = []

            for tag, role in tag2role.items():
                # keys_dict[message_id][role] = list()
                for keyval in content[1:]:
                    key, value = keyval[0], keyval[1]
                    if key == tag:
                        if value:
                            candidate = []
                            for value_str in value["strings"]: candidate.append(value_str.lower())

                            # if candidate not in keys_dict[docid][role]:
                                # keys_dict[docid][role].append(candidate)

                            new = True
                            for entity in keys_dict[docid][role]:
                                if is_subset(candidate, entity) or is_subset(entity, candidate):
                                    new = False
                            if new:
                                keys_dict[docid][role].append(candidate)

    return doc_dict, keys_dict


def generate_examples(doc_dict, keys_dict):
    examples = []
    problematic_mention_docids = []
    for docid in doc_dict:
        ex = OrderedDict([("docid", docid), ("doctext", None), ("extracts", None)])
        doc_text = doc_dict[docid]
        extracts = keys_dict[docid]
        
        # process "\n\n" and "\n"
        paragraphs = doc_text.split("\n\n")
        paragraphs_no_n = []
        for para in paragraphs:
            para = " ".join(para.split("\n"))
            paragraphs_no_n.append(para)
        doc_text_no_n = " ".join(paragraphs_no_n)
        ex["doctext"] = doc_text_no_n

        # rank the entitys and mentions within an entity (by first appearance in doctext)
        ex["extracts"] = {}
        for role in extracts:
            entity_idxs = []
            for entity in extracts[role]:
                mention_idxs = []
                for mention in entity:
                    if mention in doc_text_no_n:
                        mention_idxs.append([mention, doc_text_no_n.index(mention)])
                    else:
                        problematic_mention_docids.append([mention, docid])
                mention_idxs_sorted = sorted(mention_idxs, key=lambda m_idx: m_idx[1]) # rank mentions by mention's first appearence
                # if mention_idxs != mention_idxs_sorted: import ipdb; ipdb.set_trace()
                if mention_idxs_sorted:
                    entity_idxs.append(mention_idxs_sorted)

            try:
                entity_idxs_sorted = sorted(entity_idxs, key=lambda e_idx: e_idx[0][1]) # rank entitys by entity's first mention's first appearence
            except IndexError:
                import ipdb; ipdb.set_trace()

            # # for debug and analysis
            # if entity_idxs != entity_idxs_sorted and len(entity_idxs[0]) > 1: 
            #         print(docid, role, "\n", entity_idxs, "\n", entity_idxs_sorted, "\n")

            ex["extracts"][role] = entity_idxs_sorted
        examples.append(ex)

    return examples

if __name__=='__main__':

    # spacy
    # nlp = spacy.load("en_core_web_sm")

    for div in ["train", "dev", "test"]:
        doc_dict, keys_dict = read_files(div)
        examples = generate_examples(doc_dict, keys_dict)

        # normal written
        processed_file = "../processed/" + div + ".json"
        with open(processed_file, "w+") as f_processed:
            for ex in examples:
                f_processed.write(json.dumps(ex) + "\n")
        # pretty written
        processed_file = "../processed/pretty_" + div + ".json"
        with open(processed_file, "w+") as f_processed:
            for ex in examples:
                f_processed.write(json.dumps(ex, indent=4) + "\n")
