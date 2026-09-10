"""Small regression checks for filters that can silently discard good records."""

import unittest
import sys
from pathlib import Path

import pandas as pd

ACQUISITION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACQUISITION_ROOT))

from scripts.authentic.load_data import classify_texts, clean_source


ENGLISH = (
    "HR ANALYST Summary Experienced professional with background in Human Resources, "
    "Administrative, and Customer Service environments. Proficient in MS Excel, Word, "
    "Power Point, Share Point, Taleo, Autonomy, One Note, SAP, Visio, OrgPlus, and Outlook."
)
FRENCH = (
    "Nous recherchons un ingénieur logiciel expérimenté pour développer des applications "
    "et travailler avec notre équipe. Le candidat doit maîtriser la programmation et "
    "posséder une expérience professionnelle dans le développement de logiciels."
)


class LoadingTests(unittest.TestCase):
    def test_product_name_does_not_change_english_detection(self):
        languages = classify_texts([ENGLISH, FRENCH, ""])
        self.assertEqual([item[0] for item in languages], ["en", "fr", "unknown"])
        self.assertGreater(languages[0][1], 0.7)

    def test_english_deduplication_and_audit(self):
        raw = pd.DataFrame({
            "resume_id": ["1", "2", "3", "4", None],
            "resume_text": [ENGLISH, ENGLISH.upper(), FRENCH, "", ENGLISH],
            "source_row": range(5),
        })
        accepted, audit, counts = clean_source(raw, "resumes", 0.7)
        self.assertEqual(accepted["resume_id"].tolist(), ["1"])
        self.assertEqual(len(audit), len(raw))
        self.assertEqual(counts["duplicate_id_or_text"], 1)
        self.assertEqual(counts["empty_text"], 1)
        self.assertEqual(counts["missing_id"], 1)
        self.assertEqual(counts["non_english_or_unknown"], 1)
        self.assertEqual(raw.loc[3, "resume_text"], "")


if __name__ == "__main__":
    unittest.main()
