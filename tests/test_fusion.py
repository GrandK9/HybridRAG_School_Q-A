import unittest

from hybridrag.fusion import reciprocal_rank_fusion


class FusionTest(unittest.TestCase):
    def test_document_in_both_lists_ranks_first(self):
        fused = reciprocal_rank_fusion(
            [
                ["a", "b", "c"],
                ["b", "c", "d"],
            ],
            k=60,
        )
        self.assertEqual(fused[0][0], "b")
        self.assertGreater(fused[0][1], fused[1][1])

    def test_single_list_keeps_order(self):
        fused = reciprocal_rank_fusion([["a", "b"]], k=60)
        self.assertEqual([doc_id for doc_id, _ in fused], ["a", "b"])

    def test_duplicate_in_one_list_counts_once(self):
        fused = reciprocal_rank_fusion([["a", "a", "b"]], k=60)
        scores = dict(fused)
        self.assertAlmostEqual(scores["a"], 1.0 / 61)


if __name__ == "__main__":
    unittest.main()
