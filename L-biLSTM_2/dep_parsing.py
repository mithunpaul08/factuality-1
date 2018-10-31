from processors import ProcessorsBaseAPI
import sys
API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)

body="he can not remember leaving"
data=API.fastnlp.annotate(body)


#dependencies = [s['graphs']['stanford-collapsed'] for s in data.sentences]
#print(dependencies)

for s in data.sentences:
    # play around with the dependencies directly
    deps = s.dependencies

    # see what dependencies lead directly to the first token (i.e. token 0 is the dependent of what?)
    print(deps.incoming[0])

    # see what dependencies are originating from the first token (i.e. token 0 is the head of what?)
    print(deps.outgoing[3])
    sys.exit(1)


    g=s.graphs
    st=g["stanford-collapsed"]
    print(st)
    edges=st.edges
    print(edges[0]["destination"])
    #dest = edges["destination"]
    #print(dest)    
#src = edges["source"]
    #rel = edges["relation"]
