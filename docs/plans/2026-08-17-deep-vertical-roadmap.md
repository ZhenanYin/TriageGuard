# Deep Vertical Assurance Chain Roadmap

**Status:** Active

**Current gate:** Gate A — Correct and bounded front half

**Governing design:**
[`docs/architecture/2026-08-17-deep-vertical-assurance-chain-design.md`](../architecture/2026-08-17-deep-vertical-assurance-chain-design.md)

## 1. Why This Roadmap Exists

This is the durable navigation document for the next TriageGuard sessions. It
prevents the project from returning to case-by-case fixes that make one pull
request work without strengthening the full assurance chain.

The initial research scope is deliberately narrow: OpenMRS Core, Java-centered
security behavior, frozen Git evidence, bounded structured model calls, human
review, constrained test generation, and differential execution against the
frozen base and merge candidate.

The approved contribution is not “an LLM reviews a PR.” It is a reproducible
chain in which every model-visible fact, review decision, generated artifact,
execution result, and severity decision is attributable and independently
checkable.

## 2. Current Baseline

As of 2026-08-17:

- the existing Milestone 2 workflow reaches approved Gherkin in replay and live
  modes;
- the full test suite passed with 748 tests before this planning update;
- live PR 6456 exposed a provider request-size failure and then an
  evidence-insufficient outcome;
- the current working tree contains useful risk-stage request-size diagnostics
  and an emergency risk-only compaction experiment;
- that experiment is not the final evidence architecture because it exposes
  only a small subset of the saved context and does not apply the same boundary
  to testability and Gherkin generation;
- legitimate `M == B` snapshots are still rejected even though they mean that
  the base branch has not drifted since the PR diverged;
- the current local “grounding” validators can inspect evidence the model did
  not receive, so the term is stronger than the implementation warrants;
- Milestone 1 execution and Milestone 2 review remain separate workflows.

The local working-tree changes must be audited during Gate A. Preserve their
safe provider diagnostics, replace their risk-only compaction with the shared
evidence envelope, and do not discard unrelated user work.

## 3. Delivery Gates

### Gate A — Correct and bounded front half

**Active implementation plan:**
[`docs/plans/2026-08-17-gate-a-evidence-boundary-implementation.md`](2026-08-17-gate-a-evidence-boundary-implementation.md)

Exit conditions:

- valid OpenMRS snapshots permit `M == B` while incoherent equalities remain
  rejected;
- `M -> B` with equal revisions produces a canonical, hash-bound `unchanged`
  comparison artifact;
- risk, testability, and Gherkin stages use the same immutable
  `ModelEvidenceEnvelope` abstraction;
- citations are validated only against exact model-visible text;
- complete provider request bodies are measured before calls and fit a declared
  provider/model budget;
- no stage silently truncates evidence to make a request fit;
- bounded frozen-evidence refinement either produces a successor envelope or a
  typed evidence-insufficient outcome;
- provider and validation failures are durable, recoverable, stage-specific,
  and safe to show in the UI;
- replay, recovery, and full-suite checks pass.

### Gate B — Scoped real-PR compatibility

Write the detailed Gate B implementation plan only after Gate A measurements
are available.

Planned outcomes:

- a versioned manifest of 25–30 stratified OpenMRS Core PRs;
- a non-interactive compatibility runner that records stage reachability,
  reason codes, request sizes, evidence coverage, latency, cost, abstention,
  and human-intervention points;
- an explicit supported/unsupported behavior matrix;
- cohort-level defect triage instead of PR-specific special cases;
- GitHub Actions for compilation, Ruff, unit/integration tests, replay fixture
  verification, and artifact-schema compatibility;
- an evidence-based decision about the provider/model budget used for Gate C.

Exit conditions:

- the pilot is repeatable from a clean checkout;
- every failed PR has a stable local stage and reason code;
- supported PR classes and observed compatibility rates are documented;
- CI passes on the feature branch and pull request.

### Gate C — First complete assurance chain

Write the detailed Gate C implementation plan after Gate B identifies the
first supported OpenMRS security behavior and its environmental requirements.

Planned outcomes:

- an allowlisted OpenMRS operation catalog;
- an execution adapter that builds the exact frozen B and C revisions;
- constrained plan generation from the approved Gherkin only;
- local AST, import, symbol, data-flow, and step validation for generated
  pytest;
- identical controlled execution against B and C;
- persisted raw observations, controls, repeats, and predefined behavioral
  classification;
- a severity gate that withholds CVSS unless execution evidence is eligible.

Exit conditions:

- at least one supported Java-centered risk class completes the entire chain;
- the run is reproducible from its sealed record;
- unknown or incomplete execution cannot become a vulnerable, safe, or
  CVSS-bearing result.

### Gate D — Research evaluation

Write the detailed Gate D plan after the Gate C executable slice is stable.

Planned outcomes:

- frozen benign, known security-relevant, and controlled-mutation cohorts;
- diff-only and ungated four-role baselines;
- assurance-boundary ablations;
- expert labeling and disagreement procedures;
- compatibility, precision/recall, abstention, execution, reproducibility,
  human-effort, latency, token, and cost measurements;
- a claim-to-evidence table that constrains the paper's conclusions.

Exit conditions:

- evaluation artifacts and procedures are versioned and repeatable;
- claims distinguish operational compatibility, reviewer usefulness, and
  security accuracy;
- limitations match the measured evidence.

## 4. Session Start Protocol

Every implementation session must:

1. Read the governing design, this roadmap, and the active gate plan.
2. Inspect `git status --short`, the current branch, and recent commits before
   editing.
3. Preserve unrelated and uncommitted user changes.
4. Work on the earliest incomplete task in the active gate plan unless a newly
   discovered dependency changes the ordering.
5. Add a failing test before changing behavior.
6. Use `apply_patch` for repository edits; do not create disposable
   `apply_*.py`, `fix_*.py`, or `add_*.py` scripts in the repository.
7. Run the smallest relevant tests after each change, then the gate-level and
   full verification commands named in the plan.
8. Update the active plan's progress and decision log when facts change.
9. Commit coherent, reviewed increments locally. Do not push, merge, or delete
   branches without the user's explicit approval.

## 5. Architecture Rules for All Gates

- The frozen snapshot is the source of truth; later repository or human facts
  cannot be inserted as test evidence.
- Equality is allowed only where a documented Git role relationship permits it.
- Every comparison has an explicit status; absence and equality are not
  inferred from missing data.
- Model-visible evidence is an immutable, hashed artifact, not an incidental
  prompt dictionary.
- A local validator cannot use hidden evidence to validate a model claim.
- Provider limits are checked against the exact serialized request body before
  invocation.
- Evidence is selected as complete bounded excerpts; silent byte slicing is
  prohibited.
- Evidence insufficiency is an explicit outcome, never evidence of safety.
- Human edits create successor artifacts and invalidate incompatible downstream
  state.
- Model prose cannot establish a vulnerability or CVSS score.
- Runtime classification requires complete attributable B/C observations.
- New boundaries are extracted incrementally; the workflow coordinator should
  lose responsibilities rather than grow new domain logic.

## 6. Plan Registry

| Plan | State | Depends on | Completion evidence |
| --- | --- | --- | --- |
| Deep vertical governing design | Approved | — | Local commit `9fed604` |
| Gate A evidence boundary | Active | Governing design | Gate A exit checklist and verification log |
| Gate B compatibility pilot | Not yet written | Gate A measurements | Pilot report and CI |
| Gate C execution bridge | Not yet written | Gate B support matrix | Reproducible B/C run |
| Gate D evaluation | Not yet written | Gate C executable slice | Frozen evaluation results |

## 7. Decision Log

- **2026-08-17:** Focus on deep vertical integration for OpenMRS Core before
  repository or language breadth.
- **2026-08-17:** Keep the four logical roles M, B, H, and C, while allowing
  semantically valid equality such as `M == B`.
- **2026-08-17:** Replace stage-specific prompt payloads with one shared,
  hash-bound model-evidence envelope.
- **2026-08-17:** Treat model citations as citation-validated only; semantic
  support remains uncertain until stronger evidence or execution exists.
- **2026-08-17:** Permit only frozen-code evidence refinement in the initial
  scope; human-supplied test facts remain out of scope.
- **2026-08-17:** Make approved Gherkin an intermediate artifact, not the end of
  the research pipeline.
- **2026-08-17:** Defer detailed Gate B–D task plans until the preceding gate
  provides measurements needed to make them concrete.

## 8. Next Action

Begin Task 1 of the Gate A implementation plan: characterize and correct
snapshot-role equality and canonical unchanged comparison semantics.
