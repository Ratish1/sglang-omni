import sqlite3
import tempfile
import unittest
from pathlib import Path
from analyze_nsys_sqlite import summarize, choose_table, devices, open_database


class TestIntervals(unittest.TestCase):
    def test_overlaps_are_not_double_counted(self):
        r = summarize([(10, 30), (20, 40), (21, 25), (60, 80)], 0, 100)
        self.assertEqual(r['captured_kernel_union_ns'], 50)
        self.assertEqual(r['sum_of_clipped_kernel_durations_ns'], 64)
        self.assertEqual(r['gap_count_including_window_edges'], 3)
        self.assertEqual(r['captured_kernel_union_fraction'], .5)

    def test_empty_window(self):
        r = summarize([], 100, 200)
        self.assertEqual(r['captured_kernel_union_ns'], 0)
        self.assertEqual(r['gap_count_including_window_edges'], 1)

    def test_boundary_clipping(self):
        r = summarize([(-10, 10), (10, 20), (80, 200)], 0, 100)
        self.assertEqual(r['captured_kernel_union_ns'], 40)
        self.assertEqual(r['no_captured_kernel_ns'], 60)

    def test_full_nested(self):
        r = summarize([(-1, 101), (10, 20), (99, 103)], 0, 100)
        self.assertEqual(r['captured_kernel_union_ns'], 100)
        self.assertEqual(r['gap_count_including_window_edges'], 0)

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            summarize([], 10, 10)

    def test_unsorted(self):
        with self.assertRaises(ValueError):
            summarize([(10, 20), (0, 5)], 0, 100)

    def test_large_timestamps(self):
        n = 10**18
        r = summarize([(n+1, n+4), (n+3, n+7)], n, n+10)
        self.assertEqual(r['captured_kernel_union_ns'], 6)

    def test_no_overlap_outside_window(self):
        r = summarize([(-10, 0), (100, 110)], 0, 100)
        self.assertEqual(r['captured_kernel_count_intersecting_window'], 0)

    def test_top_gap_limit(self):
        r = summarize([(10, 20), (40, 50), (80, 90)], 0, 100, 2)
        self.assertEqual(len(r['largest_no_captured_kernel_intervals']), 2)
        self.assertEqual(r['largest_no_captured_kernel_intervals'][0]['start_ns'], 50)

    def test_schema_and_readonly(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.sqlite'
            c = sqlite3.connect(path)
            c.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER)')
            c.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?)', [(0,10,0),(5,15,0),(0,30,1)])
            c.commit()
            c.close()
            with open_database(path) as c:
                table, cols = choose_table(c, None)
                self.assertEqual(devices(c, table, cols)[1]['export_device_id'], 1)
                with self.assertRaises(sqlite3.OperationalError):
                    c.execute('CREATE TABLE bad (x INTEGER)')


if __name__ == '__main__':
    unittest.main()
