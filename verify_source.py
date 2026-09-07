from pathlib import Path
import hashlib, json
root=Path(__file__).resolve().parent
tree=json.loads((root/'tree.json').read_text())
checked=[]
for x in tree['tree']:
    if x['type']!='blob' or x['path'].startswith('data/'): continue
    b=(root/'scTM'/x['path']).read_bytes()
    assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==x['sha'], x['path']
    checked.append(x['path'])
print('Verified',len(checked),'upstream files at',tree['sha'])
