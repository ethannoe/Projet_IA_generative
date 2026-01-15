import unittest

import numpy as np

from src.utils.scoring import ScoringInputs, run_scoring_pipeline


class TestScoringUtils(unittest.TestCase):
    def test_skill_scoring_aligns_with_focus_job(self):
        user_texts = ["Analyse SQL et dashboards", "Mise en production légère"]
        blocks = [
            {"block_id": "da", "block_name": "Data Analysis", "weight": 1.2, "skills": ["SQL", "Dashboarding"]},
            {"block_id": "mlops", "block_name": "MLOps", "weight": 1.0, "skills": ["Docker"]},
        ]
        jobs = [
            {"job_name": "Data Analyst", "required_blocks": [{"block_id": "da", "weight": 1.0}]},
            {"job_name": "MLOps Engineer", "required_blocks": [{"block_id": "mlops", "weight": 1.0}]},
        ]

        # similarity_matrix shape: (num_texts, num_skills)
        similarity_matrix = np.array(
            [
                [0.82, 0.78, 0.30],  # text 0 vs SQL, Dashboarding, Docker
                [0.40, 0.35, 0.55],  # text 1
            ]
        )
        skill_mapping = [("da", "SQL"), ("da", "Dashboarding"), ("mlops", "Docker")]

        inputs = ScoringInputs(
            user_texts=user_texts,
            blocks=blocks,
            jobs=jobs,
            similarity_matrix=similarity_matrix,
            skill_mapping=skill_mapping,
        )

        scores = run_scoring_pipeline(inputs, top_k_skills=3, min_skill_score=0.3, focus_boost=0.15)
        top_job = scores["ranked_jobs"][0][0]
        self.assertEqual(top_job, "Data Analyst")

        top_skill_names = [s[1] for s in scores["top_skills"]]
        self.assertIn(top_skill_names[0], {"SQL", "Dashboarding"})

        debug_skills = scores["debug"]["skills"]
        self.assertTrue(any(s.get("boost_applied", 0) > 0 for s in debug_skills if s.get("block_id") == "da"))


if __name__ == "__main__":
    unittest.main()
