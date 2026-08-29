## What you should expect after a successful run

The final analysis writes its reproducibility bundle under the new direct-seed protocol
`OUTPUT_DIR` printed by the setup cell.

Some legacy RQ1 filenames are retained for project continuity, but RQ2 outputs and
paper-facing labels now use **seed-manipulation effect**, not exposure-only acquisition.

Key outputs include:
- `run_plan.json` — exact live prices, model panel, chosen uniform `LIMIT`, phase caps,
  and `seed_manipulation_protocol`;
- `integrity_report.json` — hard completeness/pairing checks;
- `contagion_results.csv` and `turns.csv` — structured Stage-1 results;
- `patient_zero_exposure_realization.csv` — legacy filename recording how often the
  deliberately seeded Patient Zero actually realized a successful trigger hack;
- `seed_exposure_diagnostics.csv` — separates failure to receive a source hack from
  non-adoption after receiving one;
- `control_baseline_summary.csv` — null-independent and matched honest-seed baselines;
- `contagion_funnel.csv` — PZ realization → network exposure → seed-linked adoption →
  successful post-exposure hack → operational innovation;
- `rq1_primary_paired_contrasts.csv` — primary paired intention-to-treat contrasts;
- `rq1_realized_seed_descriptive_contrasts.csv` — secondary realized-source descriptive
  analysis;
- `rq2_run_level_seed_manipulation_effects.csv` and
  `rq2_cell_seed_manipulation_effects.csv` — canonical RQ2 treatment-effect outputs;
- RQ3 eligibility, results, carryover, and sycophancy outputs;
- phase budget reports, `budget_audit.json`, `manuscript_results.json`, and figures.

### What “ready for paper results” means here

If the notebook reaches `FINAL ANALYSIS COMPLETE`, the implemented experiment has
passed code-level checks for:
- complete intended Stage-1 cell coverage and equal trials per cell;
- exact treatment/control blocking on task queue, models, graphs, and susceptible
  denominators;
- synchronous previous-round-only context;
- model-generated Patient Zero on every ticket;
- Patient Zero receiving **both** private reward-hacking examples and the private direct
  reward-hacking directive on trigger tickets;
- matched control agent 0 receiving honest examples + an honest-only private directive;
- susceptible peers never directly receiving either seed directive;
- prompt-window-aligned exposure/lineage/copying measurement;
- smoke validation of the source-to-network routing path without selecting on peer
  adoption;
- separate reporting of source realization, direct exposure, post-exposure adoption,
  successful post-exposure hacking, seed lineage, copying, and macro innovation;
- RQ3 conditional selection and history-vs-reset analysis;
- provider-side budget guards.

This protocol **does not test whether examples alone create reward hacking in Patient
Zero**. It tests propagation from an experimentally seeded reward-hacking source agent.
The source manipulation is intentionally strong so that weak downstream spread cannot be
dismissed merely as failure to create Patient Zero.

A weak or zero spread result remains valid: report source-realization rate, direct
exposure rate, conditional peer adoption, matched honest/null baselines, T80 reach
probability, and paired seed-manipulation confidence intervals.
