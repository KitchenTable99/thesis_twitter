#!/bin/bash
# download file
twarc2 hydrate download.txt temp/out.jsonl

# flatten output
twarc2 flatten temp/out.jsonl temp/flat.jsonl

# csv-ify output
twarc2 csv temp/flat.jsonl temp/flat.csv

# clean output and rewrite the proper download.txt file
python main.py
