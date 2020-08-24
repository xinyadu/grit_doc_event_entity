#!/bin/bash
# mkdir -p proc_output

set -eu

# function compose_html() {
#   tag=$1
#   table=$2
#   err=$3
#   (
#   echo "<h1>$tag keys</h1>"
#   if [[ $(cat $err | wc -c) -gt 0 ]]; then
#     echo "<h2>Warnings during processing</h2>"
#     echo "<pre>"
#     cat $err
#     echo "</pre>"
#   fi
#   echo "<h2>Keys: Left=original, Right=processed (JSON format)</h2>"
#   cat $table
#   ) > proc_output/keys_${tag}.html
# }


# cat data/TASK/CORPORA/dev/key-dev-*     | python scripts/proc_keys.py --format sidebyside 1>out.html 2>err.log
# compose_html dev out.html err.log

# cat data/TASK/CORPORA/testsets/key-tst* | python scripts/proc_keys.py --format sidebyside 1>out.html 2>err.log
# compose_html tst out.html err.log

# train
cat ../muc34/TASK/CORPORA/dev/dev-*     | python proc_texts.py > ../proc_output/doc_train

# dev
cat ../muc34/TASK/CORPORA/tst1/tst1-muc3 | python proc_texts.py > ../proc_output/doc_dev
cat ../muc34/TASK/CORPORA/tst2/tst2-muc4 | python proc_texts.py >> ../proc_output/doc_dev

# test
cat ../muc34/TASK/CORPORA/tst3/tst3-muc4 | python proc_texts.py > ../proc_output/doc_test
cat ../muc34/TASK/CORPORA/tst4/tst4-muc4 | python proc_texts.py >> ../proc_output/doc_test

