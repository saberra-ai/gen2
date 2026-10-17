"""Reject another runtime identity before any framework/signature inspection."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from verify_bounded_pnola_app import verify_app


class BoundedAppIdentityTests(unittest.TestCase):
    def test_other_profiles_and_metal_are_rejected_before_native_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            for profile, backend in [
                ('gen2-valid-mel-v1', 'cpu'),
                ('gen2-valid-mel-pool-v1', 'cpu'),
                ('gen2-valid-mel-pool-cpu-v2.extra', 'cpu'),
                ('gen2-valid-mel-pool-cpu-v3', 'cpu'),
                ('gen2-valid-mel-pool-cpu-v2', 'metal'),
            ]:
                with self.subTest(profile=profile, backend=backend), \
                        patch('verify_bounded_pnola_app.verify', return_value={
                            'runtime_profile': profile, 'backend': backend}), \
                        patch('verify_bounded_pnola_app.inspect_library') as inspect, \
                        patch('verify_bounded_pnola_app.subprocess.run') as run:
                    with self.assertRaisesRegex(ValueError, 'exact bounded CPU app profile'):
                        verify_app(Path(directory))
                    inspect.assert_not_called()
                    run.assert_not_called()
