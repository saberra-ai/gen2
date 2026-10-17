import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ggml_matrix_profile import summarize
from test_ggml_matrix_profile import fixture
import windows_matrix_profile_report as report


class MatrixProfileReportTest(unittest.TestCase):
    def profile(self):return summarize('\n'.join(fixture()),True,expected_graphs=1,expected_threads=4,expected_matrices=1)

    def test_complete_records_are_retained_once_and_replayed_from_trace(self):
        value=self.profile()
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'profile.json';record=report.save_profile(path,value)
            self.assertEqual(record['worker_matrix_records'],4)
            self.assertNotIn('cpu',json.loads(path.read_text()))
            self.assertEqual(json.loads(path.read_text())['graphs'],value['graphs'])
            self.assertIn('cpu',value)
            self.assertEqual(report.admit_saved_profile(path,record,self.profile()),record)
            with self.assertRaises(FileExistsError):report.save_profile(path,value)

    def test_changed_file_metadata_or_trace_rejects(self):
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'profile.json';value=self.profile();record=report.save_profile(path,value)
            for changed in ({**record,'worker_matrix_records':3},{**record,'file':'other.json'}):
                with self.assertRaises(ValueError):report.admit_saved_profile(path,changed,value)
            value['graphs'][0]['workers'][0]['matrices'][0]['chunks'][0]=1
            with self.assertRaises(ValueError):report.admit_saved_profile(path,record,value)
            path.write_bytes(path.read_bytes()+b' ')
            with self.assertRaises(ValueError):report.admit_saved_profile(path,record,self.profile())

    def test_byte_bound_rejects_before_creation_or_read(self):
        with tempfile.TemporaryDirectory()as folder,patch.object(report,'MAX_PROFILE_BYTES',8):
            path=Path(folder)/'profile.json'
            with self.assertRaises(ValueError):report.save_profile(path,self.profile())
            self.assertFalse(path.exists());path.write_bytes(b'x'*9)
            with self.assertRaises(ValueError):report.admit_saved_profile(path,{},self.profile())

    def test_disabled_profile_keeps_zero_records(self):
        value=summarize('',False,expected_graphs=84,expected_threads=4,expected_matrices=129)
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'profile.json';record=report.save_profile(path,value)
            self.assertEqual((record['graphs'],record['worker_matrix_records']),(0,0))
            report.admit_saved_profile(path,record,value)


if __name__=='__main__':unittest.main()
