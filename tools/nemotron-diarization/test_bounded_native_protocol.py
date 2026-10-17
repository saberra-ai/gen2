"""Exercise the real frozen delivery declarations against artifact admission."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from macos_portable_delivery import admit_delivery


class BoundedNativeProtocolTests(unittest.TestCase):
    def test_each_real_delivery_has_the_complete_external_admission_contract(self):
        protocol = json.loads(Path(__file__).with_name('macos-bounded-native-v2.protocol.json').read_text())
        for name in ['producer','native_reference']:
            with self.subTest(delivery=name),tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                declared = protocol[name]
                contract = {key:declared[key] for key in ['id','name','run_id','source_commit']}
                proof = b'fixed delivery schema control'
                archive = root / 'proof.zip'
                with zipfile.ZipFile(archive,'w') as stream: stream.writestr('proof.txt',proof)
                expected = {**contract,'zip':{'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},
                            'files':{'proof.txt':{'bytes':len(proof),'sha256':hashlib.sha256(proof).hexdigest()}}}
                api = {'id':contract['id'],'name':contract['name'],'size_in_bytes':expected['zip']['bytes'],
                       'digest':'sha256:'+expected['zip']['sha256'],
                       'workflow_run':{'id':contract['run_id'],'head_sha':contract['source_commit']}}
                metadata = root / 'metadata.json'; metadata.write_text(json.dumps(api))
                admit_delivery(archive,metadata,expected,root / 'delivery')
                self.assertEqual((root / 'delivery/proof.txt').read_bytes(),proof)
