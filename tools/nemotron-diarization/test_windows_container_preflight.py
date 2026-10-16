import unittest

from windows_container_preflight import candidate


class WindowsContainerPreflightTests(unittest.TestCase):
    def test_requires_local_windows_engine_and_matching_declared_host(self):
        version = {'Server': {'Os': 'windows'}}
        info = {'OSType': 'windows', 'Architecture': 'x86_64'}
        host = {'build': '20348'}
        self.assertTrue(candidate(version, info, host, []))
        for observed in [({}, info, host, []), (version, {'OSType': 'linux', 'Architecture': 'x86_64'}, host, []),
                         (version, info, {'build': '22631'}, []), (version, info, host, ['DOCKER_HOST']),
                         (version, {'OSType': 'windows', 'Architecture': 'arm64'}, host, [])]:
            with self.subTest(observed=observed):
                self.assertFalse(candidate(*observed))


if __name__ == '__main__':
    unittest.main()
