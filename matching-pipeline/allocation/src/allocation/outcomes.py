"""Evaluation-only access to hidden simulator state; never used by allocation."""
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd


class OutcomeOracle:
    def __init__(self, data_dir: Path, candidates: pd.DataFrame, jobs: pd.DataFrame):
        # data_dir is ground-truth-generation/data/v1.
        source = data_dir.parent.parent / 'scripts/generate_ground_truth.py'
        spec = importlib.util.spec_from_file_location('marketplace_ground_truth', source)
        self.simulator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.simulator)
        self.candidates = candidates.set_index('candidate_id')
        self.jobs = jobs.set_index('job_id')
        self.ch = pd.read_parquet(data_dir / 'candidate_hidden.parquet')
        self.jh = pd.read_parquet(data_dir / 'job_hidden.parquet')
        seeds = set(self.ch.simulation_seed) | set(self.jh.simulation_seed)
        if len(seeds) != 1:
            raise ValueError('Hidden simulator tables must use the same seed')
        self.seed = int(next(iter(seeds)))

    def reveal(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Generate even previously unmaterialized pairs with the original rules."""
        rows = []
        norm = self.simulator.norm
        for pair in pairs.itertuples(index=False):
            c, j = self.candidates.loc[pair.candidate_id], self.jobs.loc[pair.job_id]
            skill = occupation = experience = 0.0
            if c.role_family == j.role_family:
                skills = {norm(x) for x in c.skills}
                required, preferred = ({norm(x) for x in j.required_skills},
                                       {norm(x) for x in j.preferred_skills})
                skill = len(skills & required) / len(required) if required else .6
                if preferred:
                    skill = .8 * skill + .2 * len(skills & preferred) / len(preferred)
                occupation = (1.0 if c.role_track == j.role_track else .75) if c.role_type == j.role_type else .45
                experience = .75 if pd.isna(j.minimum_years_experience) else min(1, c.years_experience / max(1, j.minimum_years_experience))
            rows.append((pair.job_id, pair.candidate_id, np.round(skill, 4), occupation, np.round(experience, 4)))
        frame = pd.DataFrame(rows, columns=['job_id', 'candidate_id', 'skill_score', 'occupation_score', 'experience_score'])
        if frame.empty:
            return pd.DataFrame(columns=['job_id', 'candidate_id', 'true_success_probability', 'successful_performance'])
        return self.simulator.simulate_pair_outcomes(frame, self.ch, self.jh, self.seed)
