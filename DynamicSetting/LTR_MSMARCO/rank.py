import csv
import gzip

# Generates the queries and the indexing structure
def main():
    topicList = []
    querystring = {}
    with gzip.open("msmarco-doctrain-queries.tsv.gz", 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, querystring_of_topicid] in tsvreader:
            topicList.append(topicid)
            querystring[topicid] = querystring_of_topicid

    # In the corpus tsv, each docid occurs at offset docoffset[docid]
    docoffset = {}
    with gzip.open("msmarco-docs-lookup.tsv.gz", 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [docid, _, offset] in tsvreader:
            docoffset[docid] = int(offset)

    return topicList, querystring, docoffset


"""
getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
The content has four tab-separated strings: docid, url, title, body.
"""
def getcontent(docid, docoffset, f):
    f.seek(docoffset[docid])
    line = f.readline()
    lstLine = line.strip().split("\t")
    # returns [docid, url, title, documentText]
    return lstLine

# Fetches the next query Id
def fetch_next(iterTopicId, topicList):
    nextTopicId = topicList[topicList.index(iterTopicId) + 1]
    return nextTopicId