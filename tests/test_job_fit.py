import unittest

from src.scoring import rank_jobs


class TestJobFit(unittest.TestCase):
    def test_block_dominated_profile(self):
        # Profile: fort en Data Analysis, faible ailleurs
        block_scores = {"da": 0.85, "ml": 0.2, "nlp": 0.1}
        jobs = [
            {"job_name": "Data Analyst", "required_blocks": [{"block_id": "da", "weight": 1.0}]},
            {
                "job_name": "Data Scientist",
                "required_blocks": [
                    {"block_id": "ml", "weight": 1.0},
                    {"block_id": "da", "weight": 0.8},
                ],
            },
            {"job_name": "NLP Engineer", "required_blocks": [{"block_id": "nlp", "weight": 1.0}]},
        ]

        ranked = rank_jobs(block_scores, jobs)
        names_by_order = [r[0] for r in ranked]
        self.assertEqual(names_by_order[0], "Data Analyst")
        self.assertIn("Data Scientist", names_by_order[1:])
        self.assertIn("NLP Engineer", names_by_order[1:])
        # Ensure scores differ
        scores = [r[1] for r in ranked]
        self.assertTrue(scores[0] > scores[1] or scores[0] > scores[2])


if __name__ == "__main__":
    unittest.main()
