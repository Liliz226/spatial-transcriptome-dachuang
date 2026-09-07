import json, pathlib, urllib.request, concurrent.futures, base64
root=pathlib.Path('CODE/scTM'); root.mkdir(exist_ok=True)
t=json.load(open('CODE/tree.json'))
def fetch(x):
 p=x['path']
 if x['type']!='blob' or p.startswith('data/'): return
 target=root/p; target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(base64.b64decode(json.load(urllib.request.urlopen(x['url'],timeout=30))['content']))
 return p
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
 for p in ex.map(fetch,t['tree']):
  if p: print(p)

