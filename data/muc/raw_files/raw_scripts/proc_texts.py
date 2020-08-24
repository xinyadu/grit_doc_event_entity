"""
Split and clean up MUC article text files into JSON document objects
"""
import sys, re, json

data = sys.stdin.read()
matches = list(re.finditer(r'(DEV-\S+) *\(([^\)]*)\)', data))
has_source = bool(matches)
if not matches:
    matches = list(re.finditer(r'(TST\d+-\S+)', data))
#print matches

doc_infos = []

for match in matches:
    docid = match.group(1)
    d = {'docid':docid, 'char_start':match.end(), 'char_before':match.start()}
    if has_source:
        d['source'] = match.group(2)
    doc_infos.append(d)

for i in range(len(doc_infos)-1):
    doc_infos[i]['char_end'] = doc_infos[i+1]['char_before']
doc_infos[-1]['char_end'] = len(data)

#from pprint import pprint
#pprint(doc_infos[:5])

for d in doc_infos:
    raw_text = data[d['char_start']:d['char_end']].strip()

    # issue: there are sometimes recursive (multiple?) datelines.  we only get the first in that case.

    tag_re = r'\[[^\]]+\]'
    tags_re= '(?:%s\s+)+' % tag_re
    full_re = r'^(.*?)--\s+(%s)(.*)' % tags_re
    m = re.search(full_re, raw_text, re.DOTALL)
    if not m:
        print raw_text[:1000]
        assert False

    dateline = m.group(1).replace('\n',' ').strip()
    tags = m.group(2).replace('\n',' ')
    text = m.group(3)

    #print dateline
    #print tags
    #print text[:50].replace('\n',' ')
    #print raw_text[:500]

    assert tags.upper() == tags
    tags = re.findall(tag_re, tags)
    tags = [x.lstrip('[').rstrip(']').lower() for x in tags]

    d['dateline'] = dateline
    d['tags'] = tags

    text = text.strip()
    text = text.replace('[','(').replace(']',')')   ## should be easier for WSJPTB parsers, right...?
    #print text

    d['text'] = text

    print json.dumps(d)

    
