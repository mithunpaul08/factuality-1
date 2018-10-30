from processors import ProcessorsBaseAPI

API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)

body="this is a good project"
x=API.fastnlp.annotate(body)

for s in x.sentences:
    g=s.graphs
    st=g["stanford-collapsed"]
    print(st)
    #edges=st['edges']
    #print(edges)

    #print(edges)
    #dest = edges["destination"]
    #src = edges["source"]
    #rel = edges["relation"]
