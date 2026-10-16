import concurrent.futures as cf, datetime, hashlib, json, pathlib, urllib.request
ROOT=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0/research/windows-static-source-closure')
PIN='9e593bb18ea69cc5095e012465dcd675a822ed0d'
PORTS=['sentencepiece','abseil','protobuf','utf8-range','vcpkg-cmake','vcpkg-cmake-config']
def fetch(url):
 req=urllib.request.Request(url,headers={'User-Agent':'gen2-primary-source-research','Accept':'application/vnd.github+json' if 'api.github.com' in url else '*/*'})
 with urllib.request.urlopen(req,timeout=90) as f:return f.read(),f.url

def directory(port):
 url=f'https://api.github.com/repos/microsoft/vcpkg/contents/ports/{port}?ref={PIN}'
 body, final=fetch(url); (ROOT/'metadata').mkdir(exist_ok=True); (ROOT/'metadata'/f'{port}-contents.json').write_bytes(body)
 entries=json.loads(body); out=[]
 for e in entries:
  if e['type']!='file':raise RuntimeError(('unexpected directory',e))
  raw=f'https://raw.githubusercontent.com/microsoft/vcpkg/{PIN}/{e["path"]}'
  data,resolved=fetch(raw)
  gitsha=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
  if gitsha!=e['sha']:raise RuntimeError(('blob mismatch',e['path']))
  path=ROOT/e['path']; path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
  out.append({'path':e['path'],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha1':gitsha,'url':raw,'resolved_url':resolved})
 print(port,len(out),flush=True);return {'port':port,'directory_api_url':url,'files':out}
with cf.ThreadPoolExecutor(max_workers=4) as pool:ports=list(pool.map(directory,PORTS))
m={'schema_version':1,'retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'vcpkg_baseline':PIN,'ports':ports,'scope':'Raw Git blob bytes only; no source extraction, build, install or inference.'}
(ROOT/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
