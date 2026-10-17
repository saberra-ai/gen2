"""The long receiver's actual three delivery declarations meet the API contract."""
import json
from pathlib import Path
import tempfile
import unittest

from macos_portable_delivery import admit_delivery
import test_macos_portable_delivery as fixtures


class BoundedLongProtocolTests(unittest.TestCase):
    def test_actual_declarations_are_complete_and_admissible(self):
        protocol = json.loads(Path(__file__).with_name('macos-bounded-long.protocol.json').read_text())
        for key in ['producer', 'native', 'public']:
            with self.subTest(delivery=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive, metadata, expected = fixtures.PortableDelivery().fixture(root)
                declared = protocol[key]
                expected.update({name: declared[name] for name in ['id', 'name', 'run_id', 'source_commit']})
                api = json.loads(metadata.read_text())
                api.update(id=declared['id'], name=declared['name'],
                           workflow_run={'id': declared['run_id'], 'head_sha': declared['source_commit']})
                metadata.write_text(json.dumps(api))
                admit_delivery(archive, metadata, expected, root / 'admitted')
                self.assertEqual((root / 'admitted/nested/proof.txt').read_bytes(), b'known construction proof\n')
