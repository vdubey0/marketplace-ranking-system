import json
import sys
import unittest
from pathlib import Path

ACQUISITION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACQUISITION_ROOT))

from scripts.planning.plan_resumes import allocate_counts, classify_title, compile_rules


class RoleGroupingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = compile_rules(json.loads((ACQUISITION_ROOT / "config/role_families.json").read_text()))

    def test_related_software_titles_stay_together(self):
        for title in ["Software Engineer", "Software Engineer II", "Senior Software Engineer", "Sr. Software Engineer"]:
            with self.subTest(title=title):
                result = classify_title(title, self.rules)
                self.assertEqual(result["role_family"], "Software development")
                self.assertEqual(result["role_type"], "Software development (specialty unspecified)")

    def test_common_word_collisions(self):
        cases = {
            "Account Executive": "Sales and business development",
            "Senior Accountant": "Accounting and finance",
            "Construction Accountant": "Accounting and finance",
            "Senior Financial Reporting Analyst": "Accounting and finance",
            "Retail Front End Supervisor": "Retail",
            "Frontend Developer": "Software development",
            "Back-End Engineer": "Software development",
            "CNC Programmer": "Manufacturing and maintenance",
            "Senior Site Reliability Engineer": "IT, cloud and cybersecurity",
            "Registered Nurse (RN)": "Healthcare",
            "Senior Machine Learning Engineer": "Data and AI",
            "Electrical Engineer": "Engineering",
        }
        for title, family in cases.items():
            with self.subTest(title=title):
                self.assertEqual(classify_title(title, self.rules)["role_family"], family)

    def test_uncertainty_is_preserved(self):
        result = classify_title("Marketing Project Manager", self.rules)
        self.assertEqual(result["assignment_status"], "ambiguous")
        self.assertIsNone(result["role_family"])
        for title in ["Associate", "Team Member", "", "General Manager"]:
            with self.subTest(title=title):
                self.assertEqual(classify_title(title, self.rules)["assignment_status"], "unmatched")

    def test_whole_number_allocation(self):
        self.assertEqual(allocate_counts({"a": 5, "b": 3, "c": 2}, 10), {"a": 5, "b": 3, "c": 2})
        self.assertEqual(allocate_counts({"b": 1, "a": 1, "c": 1}, 2), {"b": 1, "a": 1, "c": 0})
        self.assertEqual(sum(allocate_counts({"a": 123, "b": 77, "c": 4}, 1000).values()), 1000)
        self.assertEqual(allocate_counts({"a": 0, "b": 1}, 2), {"a": 0, "b": 2})
        with self.assertRaises(ValueError):
            allocate_counts({}, 10)
        with self.assertRaises(ValueError):
            allocate_counts({"a": 1}, -1)


if __name__ == "__main__":
    unittest.main()
