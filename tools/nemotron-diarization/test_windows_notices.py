from pathlib import Path
import tempfile
import unittest

from windows_notices import install_supplement, verify_supplement


class WindowsNoticeTests(unittest.TestCase):
    def test_supplement_preserves_all_three_source_notices_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = install_supplement(root)
            self.assertEqual(set(record['files']), {'darts-clone-LICENSE', 'esaxx-LICENSE', 'sais-lite-NOTICE.txt'})
            verify_supplement(root, record)
            target = root / 'notices/nemo-speech/third_party/vcpkg/sentencepiece/sais-lite-NOTICE.txt'
            self.assertIn(b'Copyright (c) 2008-2009 Yuta Mori', target.read_bytes())
            original = target.read_bytes()
            target.write_bytes(original + b'\n')
            with self.assertRaises(ValueError):
                verify_supplement(root, record)
            target.write_bytes(original)
            record['files']['sais-lite-NOTICE.txt']['sha256'] = 'f' * 64
            with self.assertRaises(ValueError):
                verify_supplement(root, record)
