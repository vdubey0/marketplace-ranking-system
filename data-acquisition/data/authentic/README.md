# Downloaded English data

No generated records are included. `raw/`, `processed/`, and `reports/` are local,
ignored by Git, and can be recreated with the loading script and notebook.

## Sources acquired on 2026-09-10

| Source | Snapshot | Contents | Publisher-declared license |
| --- | --- | --- | --- |
| [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) by Sneha Anbhawal | v1; updated 2021-08-08 | 2,484 CSV rows and 2,484 PDF files; published LiveCareer resume examples | CC0: Public Domain |
| [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) by Arsh Koneru and collaborators | v13; updated 2024-08-19 | Historical job postings plus company, job-function, industry, benefit, and salary tables | CC BY-SA 4.0 |

Initial processing retained 2,481 of 2,484 resumes and 107,250 of 123,849 job
postings. Resume exclusions: 2 duplicate texts and 1 empty text. Job exclusions:
16,295 repeated IDs/texts, 222 non-English/unknown, 75 uncertain English, and
7 empty descriptions. The processed manifest and audit tables contain the counts
and per-record decisions for the actual run.

These license labels come from Kaggle's metadata, saved alongside each archive.
They are publisher declarations, not independent verification of upstream rights
or of the identity behind each resume example. Resume-example provenance is
documented in the [dataset mirror's card](https://huggingface.co/datasets/opensporks/resumes/blob/main/README.md).
Keep source attribution and the job dataset's share-alike terms with derivatives.
The resume examples are not verified unique real people. There are no observed
candidate-job matches, hiring decisions, or performance labels in these sources.

## Files

```text
raw/
  resumes/resume-dataset.zip       # Resume/Resume.csv plus original PDFs
  resumes/kaggle-metadata.json     # Metadata captured during initial setup
  jobs/linkedin-job-postings.zip   # postings.csv plus 10 supporting CSVs
  jobs/kaggle-metadata.json
processed/
  resumes.parquet                  # Accepted English resume examples
  jobs.parquet                     # Accepted English job descriptions
  resumes_audit.parquet            # One audit row for every source resume row
  jobs_audit.parquet               # One audit row for every source job row
  manifest.json                   # Source URLs, checksums, filter settings, counts
reports/
  *.png                           # Charts created by the notebook
```

Read a table with
`pd.read_parquet("data-acquisition/data/authentic/processed/resumes.parquet")`.
The archives are retained rather than expanding thousands of PDFs on disk.
The notebook reads a PDF directly from its ZIP to show the original page layout.

## Cleaning rules

- Preserve raw archives; normalize whitespace and HTML entities in processed text.
- Detect dominant language using Lingua's default accuracy, all supported languages,
  and at most 2,000 characters sampled from the start, middle, and end.
- Keep English rows with relative confidence at least 0.70 (configurable in `.env`).
  This score is not a calibrated probability. Mixed-language and short texts need
  manual review; automatic language detection is imperfect.
- Remove missing IDs, empty texts, repeated source IDs, and repeated text after
  whitespace normalization and case folding. The first qualifying copy is kept.
- Save a reason for every exclusion. Non-English rows are retained in raw archives.
- Parse supplied job timestamps as UTC and join source job-function tags by job ID.

Exact text deduplication intentionally collapses job postings that reuse identical
descriptions, including some postings in different locations. Original locations
and postings remain in the archive. Near-duplicates may remain. These counts are
unique retained documents, not verified unique candidates or hiring requisitions.

Resume `category` is the source label, not an extracted current title. Job
`source_skill_tags` are broad labels such as Engineering, not Python/SQL requirements.
Missing values remain missing. We do not invent skills, education, or experience.
Processed text is not fully anonymized. The notebook's excerpt masking is basic;
the PDF preview displays the source document. Keep raw data and notebook outputs
local unless you have reviewed what you intend to share.

The Python loader reuses archives by default and supports offline processing.
`--refresh` obtains the current upstream archive; its checksum may differ from the
snapshot documented above. Initial Kaggle metadata files are reference snapshots,
not automatically refreshed by the loader. The generated manifest's archive hashes
identify exactly which bytes were processed.
