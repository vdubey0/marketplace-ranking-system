import tempfile
import unittest
import sys
from pathlib import Path

import pandas as pd
import pymupdf

ACQUISITION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACQUISITION_ROOT))

from scripts.synthetic.generate_resume_pilot import (
    PILOT_PROFILES,
    comparable_text,
    generate,
    pilot_allocation,
    resume_text,
    validate_profile,
)


class SyntheticResumePilotTests(unittest.TestCase):
    def test_reviewed_profiles_are_consistent(self):
        for family, profile in PILOT_PROFILES.items():
            with self.subTest(family=family):
                validate_profile(profile)
                text = resume_text(profile)
                self.assertTrue(all(skill in text for skill in profile["skills"]))
                self.assertTrue(all(job["title"] in text for job in profile["experience"]))

    def test_three_record_ratio_selects_three_largest_families(self):
        counts = {
            "Healthcare": 13447,
            "Sales and business development": 7056,
            "Accounting and finance": 4798,
            "Engineering": 4493,
        }
        self.assertEqual(
            pilot_allocation(counts, 3),
            {"Healthcare": 1, "Sales and business development": 1, "Accounting and finance": 1},
        )

    def test_pdf_text_normalization(self):
        self.assertIn(comparable_text("Patient education"), comparable_text("Patient\neducation"))

    def test_generated_pilot_has_matching_files_and_selectable_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pilot"
            manifest = generate(output)
            table = pd.read_parquet(output / "resumes.parquet")
            self.assertEqual(manifest["record_count"], 3)
            self.assertEqual(len(table), 3)
            self.assertTrue(table["candidate_id"].is_unique)
            self.assertTrue(table["is_synthetic"].all())
            for row in table.itertuples():
                pdf_path = output / "pdf" / f"{row.candidate_id}.pdf"
                with pymupdf.open(pdf_path) as document:
                    self.assertEqual(len(document), 1)
                    text = " ".join(page.get_text() for page in document)
                self.assertNotIn("NOT A REAL PERSON", text)
                self.assertNotIn("fictional", text.casefold())
                self.assertIn(comparable_text(row.headline), comparable_text(text))


if __name__ == "__main__":
    unittest.main()
