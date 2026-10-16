"""Read DLL bytes as data; never loads/executes a DLL or invokes an installer."""
import ast,datetime,hashlib,importlib.util,json,pathlib,re,shutil,subprocess,sys
sys.dont_write_bytecode=True
P=pathlib.Path(__file__).parent;W=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0/tools/nemotron-diarization')
package=(P/'package_windows-input-copy.py').read_bytes();tree=ast.parse(package);crt=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='CRT' for x in n.targets))
assert hashlib.sha256(package).hexdigest()=='9e4d30d3acbd0ef21a73dff51d0f5f81ca93bea8560457524d9fcda3f3f27248', 'package_windows.py changed; use the recorded helper/input pin'
helper=(P/'windows_pe-inspection-copy.py').read_bytes();assert hashlib.sha256(helper).hexdigest()=='d194378b4a666444c350f4f4896debcee649568714ae51fc7797844f1cf4d52c', 'windows_pe.py changed; use the recorded helper pin'
(P/'windows_pe-inspection-copy.py').write_bytes(helper)
spec=importlib.util.spec_from_file_location('cached_windows_pe',P/'windows_pe-inspection-copy.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
selected=sorted(crt|{'vcomp140.dll'});rows=[];logs=P/'minimum-pe-logs';logs.mkdir(exist_ok=True)
for name in selected:
 path=P/'minimum-cab'/(name+'_amd64');data=path.read_bytes();info=module.inspect_pe(path)
 assert info['machine']=='AMD64' and info['format']=='PE32+' and info['is_dll'] and info['export_name']==name
 raw=subprocess.check_output(['/opt/homebrew/bin/7z','l','-slt','-tPE',str(path)],text=True);log=logs/(name+'.txt');log.write_text(raw)
 filever=re.search(r'^Comment = FileVersion: (.+)$',raw,re.M).group(1);prodver=re.search(r'^ProductVersion: (.+)$',raw,re.M).group(1)
 # Export names are large and unrelated to this limited dependency query; retain their digest/count.
 exports=info.pop('exports');info['export_names_json_sha256']=hashlib.sha256(json.dumps(exports,separators=(',',':')).encode()).hexdigest()
 rows.append({'runtime_name':name,'cab_logical_name':path.name,'selected_for':'package_windows.CRT' if name in crt else 'chat OpenMP candidate','path':str(path.relative_to(P)),'file_version_static_string':filever,'product_version_static_string':prodver,'version_method':'Existing p7zip PE resource listing; not Windows version API','pe_listing':str(log.relative_to(P)),'pe_listing_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),**info})
inventory=[]
for f in sorted((P/'minimum-cab').iterdir()):
 b=f.read_bytes();inventory.append({'cab_logical_name':f.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
meta={'schema_version':1,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'installer_sha256':'843068991daaa1f73ad9f6239bce4d0f6a07a51f18c37ea2a867e9beca71295c','burn_payload':'a4','burn_logical_path':'packages\\vcRuntimeMinimum_amd64\\cab1.cab','cab_bytes':(P/'attached/a4').stat().st_size,'cab_sha256':hashlib.sha256((P/'attached/a4').read_bytes()).hexdigest(),'source_helpers':{'windows_pe.py_sha256':hashlib.sha256(helper).hexdigest(),'package_windows.py_sha256':hashlib.sha256(package).hexdigest(),'crt_names':sorted(crt)},'cab_inventory':inventory,'selected_dlls':rows,'dlls_executed':False,'authenticode_verified':False,'actual_install_paths_verified':False,'scope':'Byte hashes and static PE identity/version strings/imports only. CAB file keys are not MSI destination paths. No Windows loading, signature trust, installation, full API-set resolution or runtime closure test.'}
(P/'minimum-x64-cab-inspection.json').write_text(json.dumps(meta,indent=2)+'\n')
for r in rows:print(r['runtime_name'],r['bytes'],r['sha256'],r['file_version_static_string'],'imports',r['imports'],'delay',r['delay_imports'],'forward',r['forwarder_dependencies'],'bound',r['bound_imports_present'])
