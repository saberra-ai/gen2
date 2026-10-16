from pathlib import Path
import subprocess,json,sys
r=Path(__file__).parent
env={'PATH':'/usr/bin:/bin'}
for minutes in [30,60]:
 for kind in ['gen2','native']:
  base=['/usr/bin/time','-l','/usr/bin/sandbox-exec','-f',str(r/'offline.sb')]
  if kind=='gen2':
   cmd=base+[str(r/'nemotron_stream'),str(r/'assets'),str(r/'nemo/lib/libnemo_speech_asr_c.dylib'),str(r/'assets/ami_en2002d_2132.wav'),str(minutes),str(r/f'gen2-{minutes}m.f32')]
  else:
   cmd=base+[str(r/'native_stream_probe'),str(r/'assets/Nemotron-3-Diarization.q8_0.gguf'),str(r/'assets/ami_en2002d_2132.wav'),str(minutes),'-1',str(r/f'native-{minutes}m.json')]
  (r/f'{kind}-{minutes}m.command.json').write_text(json.dumps({'command':cmd,'environment':env},indent=2)+'\n')
  print(f'START {kind} {minutes}m',flush=True)
  out=r/f'{kind}-{minutes}m.json' if kind=='gen2' else r/f'{kind}-{minutes}m.stdout.log'
  with out.open('w') as stdout,(r/f'{kind}-{minutes}m.time.log').open('w') as stderr:
   subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr,check=True)
  print(f'PASS {kind} {minutes}m',flush=True)
 subprocess.run([sys.executable,str(r/'compare_stream.py'),'--gen2',str(r/f'gen2-{minutes}m.json'),'--native',str(r/f'native-{minutes}m.json'),'--gen2-probabilities',str(r/f'gen2-{minutes}m.f32'),'--output',str(r/f'comparison-{minutes}m.json')],check=True)
 print((r/f'comparison-{minutes}m.json').read_text(),flush=True)
