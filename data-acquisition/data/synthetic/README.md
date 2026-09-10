# Synthetic data

Synthetic records are kept separate from downloaded and processed source data.

```text
data-acquisition/data/synthetic/
  README.md
  generated/
    role_catalog/
      v1/
        requests.jsonl
        request_sources.jsonl
        role_archetypes.jsonl
        *_manifest.json
    candidate_specs/
      v1/
        candidate_specs.jsonl
        allocation.json
        manifest.json
    resumes/
      pilot_v1/
        profiles.jsonl
        resumes.parquet
        allocation.json
        manifest.json
        text/
        pdf/
      v1/
        requests/
        responses/
        profiles/
        text/
        pdf/
        manifests/
        resumes.parquet
```

Future synthetic stages should get separate directories under `generated/`, for
example `relevance/`, `outcomes/`, and `marketplace_events/`. Generated files are
ignored by Git because they can be recreated by scripts. The scripts, templates,
configuration, seeds, and documentation are tracked.

Resume text and PDFs omit generation labels so those words cannot influence
matching experiments. Synthetic provenance remains in the structured profile,
manifest, and `is_synthetic` fields. Preserve those fields when combining data.

The role catalog is grounded in several job descriptions per role. Candidate
specifications are generated deterministically from the catalog and observed role
ratios. OpenAI writes prose from those locked specifications; Python supplies the
skills, dates, titles, invented employers, education, and certifications shown in
the final documents. Batch request and response files are retained for auditing.
