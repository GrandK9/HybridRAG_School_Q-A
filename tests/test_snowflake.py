import unittest

from hybridrag.snowflake import Snowflake


class SnowflakeTest(unittest.TestCase):
    def test_ids_are_unique_positive_int64(self):
        generator = Snowflake(worker_id=3)
        ids = [generator.next_id() for _ in range(200)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(0 < item < 2**63 for item in ids))
        self.assertEqual(ids, sorted(ids))


if __name__ == "__main__":
    unittest.main()
