from processors import ProcessorsBaseAPI

self.API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)
doc2 = API.fastnlp.annotate(body)
logger.debug(doc2)

body="this is a good project"
x=API.fastnlp.annotate(body)

for s in x.sentences:
    print(s.graphs["stanford-collapsed"]
    dest = edges["destination"]
    src = edges["source"]
    rel = edges["relation"]