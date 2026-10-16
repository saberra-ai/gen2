import pathlib,socket,sys
for name in sys.argv[1:]:
 try:
  with pathlib.Path(name).open("rb")as stream: stream.read(1)
 except PermissionError: pass
 else: raise RuntimeError("expected denied read: "+name)
try: socket.create_connection(("127.0.0.1",9),timeout=1)
except PermissionError: pass
else: raise RuntimeError("network was not denied")
print("all requested reads and loopback denied")
