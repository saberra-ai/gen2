import copy
import unittest
from cpu_thread_metrics import used_dependencies,validate_self_test,validate_failure_control


class CpuThreadMetricsTests(unittest.TestCase):
    def test_lifetime_control_rejects_growth_and_missing_attempts(self):
        value = {'complete':True,'backend_lifetimes':22,'model_constructed':False,'frontend_executed':False,
                 'thread_growth_allowance':2,'warmed_thread_baseline':1,
                 'cases':[{'threads':threads,'observations':[{'attempt':attempt,'immediate':4,'settled':1,'requested_wait_ms':10}
                          for attempt in range(1,11)]} for threads in [4,3]]}
        self.assertEqual(validate_self_test(value),value)
        partial = {**value,'complete':False,'phase':'injected-failure-control'}
        self.assertEqual(validate_failure_control(partial),partial)
        with self.assertRaises(ValueError): validate_self_test(partial)
        with self.assertRaises(ValueError): validate_failure_control(value)
        bad = copy.deepcopy(value)
        bad['cases'][1]['observations'][9]['settled'] = 4
        with self.assertRaises(ValueError): validate_self_test(bad)
        bad = copy.deepcopy(value)
        bad['cases'][1]['observations'].pop()
        with self.assertRaises(ValueError): validate_self_test(bad)

    def test_dependency_continuations_and_escaped_spaces(self):
        self.assertEqual(used_dependencies('/out/probe.o: /source/a.cpp \\\n /source/My\\ File.h\n'),
                         {'/source/a.cpp','/source/My File.h'})
        with self.assertRaises(ValueError): used_dependencies('probe.o: relative.h')
