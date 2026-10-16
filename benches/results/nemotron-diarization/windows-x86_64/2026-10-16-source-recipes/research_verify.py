import concurrent.futures as cf,datetime,hashlib,json,pathlib,re,tarfile,urllib.request
ROOT=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0/research/windows-static-source-closure')
REC=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0/platform-ci/windows-vcpkg-38890986137')
m=json.loads((ROOT/'manifest.json').read_text()); files={x['path']:x for p in m['ports'] for x in p['files']};checks=[]
for d in sorted((REC/'metadata').iterdir()):
 sp=next(d.rglob('vcpkg.spdx.json')); abi=next(d.rglob('vcpkg_abi_info.txt')); sj=json.loads(sp.read_text()); port=sp.parent.name
 abifields={line.split(' ',1)[0]:line.split(' ',1)[1] for line in abi.read_text().splitlines() if ' ' in line}; r={'package':d.name,'abi_info_sha256':hashlib.sha256(abi.read_bytes()).hexdigest(),'spdx_port_files':[],'abi_port_fields':[]}
 expected={f'ports/{port}/{x["fileName"].removeprefix("./")}':x for x in sj['files'] if x['SPDXID'].startswith('SPDXRef-port-file-')}
 actual={k for k in files if k.startswith(f'ports/{port}/')}
 if actual!=set(expected):raise RuntimeError(('port inventory mismatch',port,actual-set(expected),set(expected)-actual))
 for path,x in expected.items():
  local=files[path]; sha=next(c['checksumValue'] for c in x['checksums'] if c['algorithm']=='SHA256')
  if sha!=local['sha256']:raise RuntimeError(('SPDX mismatch',path))
  rel=path.removeprefix(f'ports/{port}/')
  if rel not in abifields:raise RuntimeError(('missing ABI portfield',d.name,rel))
  if abifields[rel]!=sha:raise RuntimeError(('ABI mismatch',d.name,rel))
  r['spdx_port_files'].append({'path':path,'sha256':sha,'match':True});r['abi_port_fields'].append({'field':rel,'sha256':sha,'match':True})
 checks.append(r)
m['package_port_verification']=checks
sources=[('sentencepiece','google/sentencepiece','v0.2.1'),('abseil','abseil/abseil-cpp','20260107.1'),('protobuf','protocolbuffers/protobuf','v33.4')]
def acquire(t):
 port,repo,ref=t;recipe=(ROOT/f'ports/{port}/portfile.cmake').read_text();expected=re.search(r'SHA512\s+([a-f0-9]{128})',recipe).group(1)
 url=f'https://github.com/{repo}/archive/{ref}.tar.gz'; dest=ROOT/'sources'/f'{repo.replace("/","-")}-{ref}.tar.gz';dest.parent.mkdir(exist_ok=True)
 req=urllib.request.Request(url,headers={'User-Agent':'gen2-primary-source-research'})
 try:
  with urllib.request.urlopen(req,timeout=120) as f:
   resolved=f.url
   with dest.open('wb') as out:
    while b:=f.read(1024*1024):out.write(b)
 except urllib.error.HTTPError as e:
  if e.code!=403:raise
  url=f'https://codeload.github.com/{repo}/tar.gz/{ref}'
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'gen2-primary-source-research'}),timeout=120) as f:
   resolved=f.url
   with dest.open('wb') as out:
    while b:=f.read(1024*1024):out.write(b)
 data=dest.read_bytes();actual=hashlib.sha512(data).hexdigest()
 if actual!=expected:raise RuntimeError(('source archive mismatch',port,actual,expected))
 out={'ports':[port,'utf8-range'] if port=='protobuf' else [port],'repository':repo,'ref':ref,'recipe_url':f'https://github.com/microsoft/vcpkg/blob/{m["vcpkg_baseline"]}/ports/{port}/portfile.cmake','recipe_download_url':f'https://github.com/{repo}/archive/{ref}.tar.gz','requested_url':url,'resolved_url':resolved,'path':str(dest.relative_to(ROOT)),'bytes':len(data),'sha512':actual,'sha256':hashlib.sha256(data).hexdigest(),'recipe_sha512_match':True}
 print('SOURCE',port,len(data),actual,flush=True);return out
with cf.ThreadPoolExecutor(max_workers=3) as pool: m['upstream_sources']=list(pool.map(acquire,sources))
# baseline root LICENSE used by the vcpkg-cmake recipe (not inside that port directory)
url=f'https://raw.githubusercontent.com/microsoft/vcpkg/{m["vcpkg_baseline"]}/LICENSE.txt'
with urllib.request.urlopen(url,timeout=30) as f:data=f.read()
p=ROOT/'vcpkg-LICENSE.txt';p.write_bytes(data)
m['vcpkg_root_license']={'url':url,'path':p.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
m['verified_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();m['scope']='Six pinned port directories: Git blob SHA1, raw SHA256, all nine SPDX port-file and ABI port fields checked. Three upstream archive raw SHA512s checked. No source extracted or executed; no native-link membership claim.'
(ROOT/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('VERIFIED',len(files),'unique port files',sum(len(x['spdx_port_files']) for x in checks),'SPDX checks',sum(len(x['abi_port_fields']) for x in checks),'ABI port checks',flush=True)
