"""
Unit testing of recorder
========================
"""


import unittest

from kingpin import recorder


class TestRecorder(unittest.TestCase):

    def test_record(self):
        r1 = recorder.Record(1, 10)
        r2 = recorder.Record(4, 10)
        r = r1 + r2
        self.assertEqual(r.efficiency, 0.2)

    def test_recorder(self):
        r1 = recorder.Recorder()
        r2 = recorder.Recorder()
        r1["e"] = recorder.Record(1, 10)
        r2["e"] = recorder.Record(4, 10)
        r2["f"] = recorder.Record(7, 10)
        r = r1 + r2
        self.assertEqual(str(r), 'e = 0.200. f = 0.412')


if __name__ == '__main__':
    unittest.main()
