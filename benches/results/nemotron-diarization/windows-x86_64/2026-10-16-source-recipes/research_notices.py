import hashlib,json,pathlib,tarfile
R=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0/research/windows-static-source-closure'); C=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0/platform-ci/windows-vcpkg-38890986137/metadata');m=json.loads((R/'manifest.json').read_text());notices=[]; inventories=[]
for src in m['upstream_sources']:
 with tarfile.open(R/src['path'],'r:gz') as t:
  members=t.getmembers();prefix=members[0].name.split('/')[0]; paths={x.name:x for x in members}
  inventory=[{'path':x.name.removeprefix(prefix+'/'),'bytes':x.size} for x in members if x.isfile() and any(z in pathlib.PurePosixPath(x.name).name.lower() for z in ['license','copying','notice','authors'])]
  inventories.append({'archive':src['path'],'license_notice_named_files':inventory})
  for port in src['ports']:
   rel='third_party/utf8_range/LICENSE' if port=='utf8-range' else 'LICENSE'; member=paths[prefix+'/'+rel];data=t.extractfile(member).read();sha=hashlib.sha256(data).hexdigest();comparisons=[]
   for d in C.glob(port+'-*'):
    files=list(d.glob('share/'+port+'/copyright'))
    for f in files:
     installed=f.read_bytes(); comparisons.append({'package':d.name,'copyright_bytes':len(installed),'copyright_sha256':hashlib.sha256(installed).hexdigest(),'source_notice_exact_match':installed==data})
   notices.append({'port':port,'source_archive':src['path'],'source_member':member.name,'bytes':len(data),'sha256':sha,'cached_copyright_comparisons':comparisons})
   print(port,rel,len(data),sha,comparisons)
for port,source in [('vcpkg-cmake',R/'vcpkg-LICENSE.txt'),('vcpkg-cmake-config',R/'ports/vcpkg-cmake-config/copyright')]:
 data=source.read_bytes();f=C/(port+'-x64-windows')/'share'/port/'copyright';inst=f.read_bytes();notices.append({'port':port,'source_file':str(source.relative_to(R)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'cached_copyright_comparisons':[{'package':port+'-x64-windows','copyright_bytes':len(inst),'copyright_sha256':hashlib.sha256(inst).hexdigest(),'source_notice_exact_match':inst==data}]});print(port,'exact',inst==data)
m['notice_source_comparisons']=notices;m['source_license_notice_inventory']=inventories;m['notice_scope']='Only recipe-selected notices were byte-compared; license/notice-named file inventory is discovery metadata, not complete attribution audit or native-link membership proof. Tar entries were read without extraction to filesystem.'
(R/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print('INVENTORIES',[(x['archive'],len(x['license_notice_named_files'])) for x in inventories])
