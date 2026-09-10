import sys
import tempfile
import unittest
from pathlib import Path

import pymupdf

ACQUISITION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACQUISITION_ROOT))

from scripts.synthetic.generate_candidate_specs import generate
from scripts.synthetic.io_utils import read_jsonl, write_jsonl
from scripts.synthetic.models import CandidateSpec, ResumeWriting, RoleArchetype
from scripts.synthetic.render_resumes import materialize, normalized_pdf_text, request_body, validate_writing


ARCHETYPE = {
    "role_family": "Administration and office support",
    "role_type": "Office administration",
    "tracks": [{
        "track_name": "General office administration",
        "target_titles": ["Administrative Assistant", "Office Coordinator", "Office Administrator"],
        "core_skills": ["Calendar management", "Meeting coordination", "Records management", "Document preparation", "Data entry"],
        "optional_skills": ["Travel coordination", "Expense reporting", "Office supplies", "Visitor reception"],
        "responsibilities": ["Maintain calendars", "Prepare documents", "Coordinate meetings", "Organize records", "Route inquiries", "Order supplies"],
        "education_options": ["Associate degree in business administration", "Coursework in office administration"],
        "certifications": ["Certified Administrative Professional"],
        "previous_titles": ["Receptionist", "Administrative Clerk"],
        "industries": ["Professional services", "Education", "Healthcare"],
    }],
    "source_job_ids": ["1", "2", "3"],
}


class ResumeGeneratorTests(unittest.TestCase):
    def test_pdf_text_normalizes_ligatures_and_hyphenated_line_wraps(self):
        extracted = "oﬀer coordination · full-\ncycle recruiting"
        normalized = normalized_pdf_text(extracted)
        self.assertIn("offer coordination", normalized)
        self.assertIn("full-cycle recruiting", normalized)

    def test_role_archetype_and_request_schema(self):
        data = dict(ARCHETYPE)
        data.pop("source_job_ids")
        archetype = RoleArchetype.model_validate(data)
        track = archetype.tracks[0]
        body = request_body("test-model", CandidateSpec.model_validate({
            "candidate_id": "syn_test_1", "generator_version": "test", "seed": 1,
            "role_family": archetype.role_family, "role_type": archetype.role_type,
            "role_track": track.track_name, "target_title": track.target_titles[0],
            "seniority": "entry", "fit_profile": "partial",
            "years_experience": 1, "skills": track.core_skills[:4],
            "experience": [{"experience_id": "exp-1", "title": track.target_titles[0],
                            "company": "Test Group", "industry": track.industries[0],
                            "start": "2025-09", "end": "2026-08", "responsibilities": track.responsibilities[:2]}],
            "education": [{"credential": track.education_options[0], "institution": "Test College", "graduation_year": 2025}],
            "certifications": [], "source_job_ids": ["1"],
        }))
        self.assertFalse(body["store"])
        self.assertTrue(body["text"]["format"]["strict"])

    def test_candidate_specs_are_deterministic_and_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.jsonl"
            write_jsonl(catalog, [ARCHETYPE])
            first = generate(catalog, root / "one", 20, 17)
            second = generate(catalog, root / "two", 20, 17)
            specs_one = (root / "one/candidate_specs.jsonl").read_text()
            specs_two = (root / "two/candidate_specs.jsonl").read_text()
            self.assertEqual(specs_one, specs_two)
            self.assertEqual(first["record_count"], 20)
            for raw in read_jsonl(root / "one/candidate_specs.jsonl"):
                spec = CandidateSpec.model_validate(raw)
                self.assertTrue(all("fictional" not in job.company.casefold() for job in spec.experience))
                self.assertTrue(all("fictional" not in cert.casefold() for cert in spec.certifications))
                for newer, older in zip(spec.experience, spec.experience[1:]):
                    self.assertLess(older.end, newer.start)

    def test_materialized_resume_omits_generation_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.jsonl"
            write_jsonl(catalog, [ARCHETYPE])
            generate(catalog, root / "specs", 1, 19)
            specs_path = root / "specs/candidate_specs.jsonl"
            spec = CandidateSpec.model_validate(read_jsonl(specs_path)[0])
            writing = ResumeWriting.model_validate({
                "candidate_id": spec.candidate_id,
                "professional_summary": "Organized administrative professional experienced in dependable office support. Skilled in maintaining accurate records and coordinating daily work.",
                "experience": [{"experience_id": job.experience_id,
                                "bullets": [f"Supported {task.lower()}." for task in job.responsibilities[:2]]}
                               for job in spec.experience],
            })
            validate_writing(spec, writing)
            manifest = materialize(specs_path, root / "resumes", [writing])
            self.assertEqual(manifest["record_count"], 1)
            pdf = root / f"resumes/pdf/{spec.candidate_id}.pdf"
            with pymupdf.open(pdf) as document:
                text = " ".join(page.get_text() for page in document)
            self.assertNotIn("NOT A REAL PERSON", text)
            self.assertNotIn("fictional", text.casefold())
            self.assertIn(spec.target_title.casefold(), text.casefold())

    def test_source_backed_numbers_are_allowed_but_invented_metrics_are_rejected(self):
        data = dict(ARCHETYPE)
        data.pop("source_job_ids")
        track = RoleArchetype.model_validate(data).tracks[0]
        spec = CandidateSpec.model_validate({
            "candidate_id": "syn_test_numbers", "generator_version": "test", "seed": 1,
            "role_family": "IT", "role_type": "Networks", "role_track": track.track_name,
            "target_title": "Network Engineer", "seniority": "mid", "fit_profile": "strong",
            "years_experience": 4, "skills": ["Layer 2 troubleshooting", "TCP/IP", "Routing", "Switching"],
            "experience": [{"experience_id": "exp-1", "title": "Network Engineer",
                            "company": "Test Group", "industry": "Technology",
                            "start": "2022-01", "end": "2025-12",
                            "responsibilities": ["Resolve L2 and L3 incidents", "Maintain network diagrams"]}],
            "education": [{"credential": "Associate degree", "institution": "Test College",
                           "graduation_year": 2021}],
            "certifications": [], "source_job_ids": ["1"],
        })
        base = {
            "candidate_id": spec.candidate_id,
            "professional_summary": "Network professional experienced in troubleshooting. Skilled in routing and switching.",
            "experience": [{"experience_id": "exp-1", "bullets": [
                "Resolved Layer 2 and Layer 3 incidents.", "Maintained network diagrams.",
            ]}],
        }
        validate_writing(spec, ResumeWriting.model_validate(base))
        base["experience"][0]["bullets"] = ["Improved uptime by 30%.", "Maintained network diagrams."]
        with self.assertRaisesRegex(ValueError, "30%"):
            validate_writing(spec, ResumeWriting.model_validate(base))


if __name__ == "__main__":
    unittest.main()
