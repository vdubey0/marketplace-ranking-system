# Which synthetic resumes should we create?

This is a draft generation plan based on our downloaded English job descriptions.
No synthetic resumes or PDFs have been created by this step.

## What was counted

- **107,250** cleaned job descriptions, with **69,667** distinct original titles.
- **65,439 (61.0%)** jobs matched exactly one role family.
- **1,721** jobs matched multiple families and need review.
- **40,090** jobs did not match a rule and need review.
- **27** assigned role families, divided into **48** resume role types.

## Proposed first batch: 1,000 resumes

Allocate resumes in proportion to the jobs assigned to each family. The quantities
sum to exactly 1,000. This is a starting mix for this dataset, not a claim
about the current labor market. The percentage column uses all 107,250 jobs;
the allocation uses only the 65,439 jobs assigned to one family.

| Resume family | Jobs | Share of all jobs | Proposed resumes | Actual title examples |
| --- | --- | --- | --- | --- |
| Healthcare | 13,447 | 12.54% | 205 | Registered Nurse / Registered Nurse - RN - LTAC / Patient Care Technician |
| Sales and business development | 7,056 | 6.58% | 108 | Salesperson / Account Executive / Account Manager |
| Accounting and finance | 4,798 | 4.47% | 73 | Senior Accountant / Staff Accountant / Controller |
| Engineering | 4,493 | 4.19% | 69 | Electrical Engineer / Project Engineer / Manufacturing Engineer |
| Project, product and business operations | 4,331 | 4.04% | 66 | Project Manager / Business Analyst / Operations Manager |
| Logistics and transport | 3,554 | 3.31% | 54 | Package Handler - Part Time (Warehouse like) / Warehouse Associate / Store Driver |
| Manufacturing and maintenance | 3,111 | 2.90% | 48 | Maintenance Technician / Production Supervisor / Service Technician |
| Software development | 2,791 | 2.60% | 43 | Senior Software Engineer / Software Engineer / Full Stack Engineer |
| IT, cloud and cybersecurity | 2,756 | 2.57% | 42 | Network Engineer / DevOps Engineer / Technical Support Specialist |
| Retail | 2,151 | 2.01% | 33 | Store Manager / Assistant Store Manager / Retail Sales Associate |
| Marketing and communications | 1,856 | 1.73% | 28 | Marketing Manager / Marketing Coordinator / Marketing Intern |
| Legal and compliance | 1,716 | 1.60% | 26 | Associate Attorney / Paralegal / Attorney |
| Data and AI | 1,602 | 1.49% | 24 | Data Analyst / Data Engineer / Data Scientist |
| Administration and office support | 1,557 | 1.45% | 24 | Administrative Assistant / Executive Assistant / Receptionist |
| Construction and skilled trades | 1,524 | 1.42% | 23 | Construction Project Manager / Construction Superintendent / Superintendent |
| Hospitality and food service | 1,437 | 1.34% | 22 | Cook / Housekeeper / Restaurant Manager |
| HR and recruiting | 1,382 | 1.29% | 21 | Human Resources Generalist / Human Resources Manager / Recruiter |
| Banking and insurance | 1,317 | 1.23% | 20 | Teller / Mortgage Loan Officer / Relationship Banker |
| Customer service | 1,257 | 1.17% | 19 | Customer Service Representative / Customer Success Manager / Customer Service Representative, Full or Part Time |
| Education and training | 844 | 0.79% | 13 | Instructional Designer / Special Education Teacher / Teacher |
| Design and creative media | 719 | 0.67% | 11 | Graphic Designer / Technical Writer / User Experience Designer |
| Science, research and agriculture | 477 | 0.44% | 7 | Laboratory Technician / Chemist / Research Assistant |
| Safety and protective services | 443 | 0.41% | 7 | Safety Manager / Security Officer / EHS Specialist |
| Automotive | 366 | 0.34% | 6 | Auto Detailer / Auto Glass Installation Technician Trainee / Automotive Technician |
| Property and real estate | 232 | 0.22% | 4 | Property Manager / Assistant Property Manager / Real Estate Agent |
| Fitness and recreation | 164 | 0.15% | 3 | Personal Trainer / Athletic Trainer / Lifeguard |
| Aviation | 58 | 0.05% | 1 | Aircraft Mechanic / Aircraft Technician / Pilot in Command Phenom 300 |

## How the grouping works

Rules match words in titles, not descriptions. Seniority words and original titles
are preserved. For example, “Software Engineer II” and “Senior Software Engineer”
belong to Software development. A generic software title does not tell us whether
someone should be a backend, frontend, or mobile developer; that specialty remains
unspecified unless the title says so. Description review is the next step.

Within a family, specific role rules run before generic ones. Across families,
multiple matches are flagged instead of resolved by guessing. Every job has its
original title, rule matches, and decision in `job_role_mapping.parquet`.

These are project-specific rules, not verified occupation labels. Do not use this
mapping as independent retrieval ground truth or treat it as a measured accuracy
result. Review the sample and unresolved titles before a large generation run.

## What the collected resumes contribute

The resume examples can guide wording and career-history structure. Their broad
source categories do not reliably identify each detailed role. The CSV includes
related source categories for reference, not confirmed matching candidates. IT
examples are referenced by more than one family, so those reference counts must
not be added together. No dedicated source category means unknown coverage, not
proof that no relevant resume exists.

## Experience levels

`source_seniority_counts.csv` preserves each family's supplied job experience levels,
including Unknown. These labels have not been converted to candidate years of
experience. Title words such as senior and manager can disagree with the supplied
labels; inspect the description before choosing candidate experience requirements.

## Files to inspect

- `data/authentic/processed/job_role_mapping.parquet`: one decision per input job.
- `data/authentic/reports/role_families/family_counts.csv`: family counts and proposed quantities.
- `data/authentic/reports/role_families/resume_generation_plan.csv`: finer role types, quantities,
  actual title examples, and job IDs to inspect before generating backgrounds.
- `data/authentic/reports/role_families/titles_to_review.csv`: ambiguous/unmatched titles, most frequent first.
- `data/authentic/reports/role_families/assignment_audit_sample.csv`: up to eight jobs per family
  for manual checking; reviewer fields start empty.
- `data/authentic/reports/role_families/manifest.json`: source hashes and reproducibility details.

Next: review uncertain titles, extract typical skills from descriptions for the
chosen roles, then create a small sample of fictional profiles and resumes. Use
multiple source jobs to understand a role; do not tailor each candidate to one job.
