# Deep Vertical Assurance Chain Design

**Status:** Approved governing design as of 2026-08-17.

**Supersedes for future planning:** The earlier Milestone 2 evidence-refinement
plan remains an implementation record, but its endpoint at approved Gherkin is
no longer the project endpoint. Future work must follow this design's complete
assurance chain before broadening repository or language scope.

## 1. Purpose

TriageGuard will focus on one narrow but complete research contribution:

> Construct a reproducible assurance chain that turns an unconfirmed security
> risk in an OpenMRS Core pull request into differential runtime evidence, and
> permit a provisional severity result only when every intermediate artifact is
> attributable, current, bounded, human-gated where required, and locally
> validated within its declared limits.

TriageGuard does not ask an LLM to decide whether a pull request is vulnerable.
The LLM proposes structured, explicitly uncertain artifacts. Deterministic
software, human review, and ultimately differential execution determine which
claims may advance.

## 2. Fixed Initial Scope

The first complete vertical slice supports:

- the `openmrs/openmrs-core` repository;
- open pull requests targeting the default branch;
- reproducible Git objects for the pull-request head and merge preview;
- Java-centered authorization and security behavior that can be represented by
  approved evidence and execution adapters;
- bounded Groq model calls through strict structured-output contracts;
- isolated comparison of the frozen base and merge-candidate revisions.

Changes outside this scope must produce an explicit unsupported or
evidence-insufficient outcome. They must not be treated as safe.

Arbitrary repositories, arbitrary languages, human-supplied test facts, and
fully autonomous approval are not part of the first complete vertical slice.

## 3. Governing Assurance Chain

```text
Freeze M / B / H / C roles
        ↓
Create M→H, B→C, and M→B comparison artifacts
        ↓
Select and hash the exact model-visible evidence
        ↓
Generate unconfirmed risk hypotheses
        ↓
Validate evidence references and disclose semantic uncertainty
        ↓
Human selects and reviews one hypothesis
        ↓
Assess structural testability from frozen evidence
        ↓
Generate and validate evidence-bound Gherkin
        ↓
Generate and locally validate a constrained test plan and pytest
        ↓
Execute the same approved test against frozen B and C environments
        ↓
Persist raw observations and classify the behavioral transition
        ↓
Allow provisional CVSS only when runtime evidence is eligible
```

Every arrow is a recorded transformation. Every transformation identifies its
exact inputs, outputs, hashes, versions, bounds, reason code, and time.

## 4. Snapshot and Comparison Semantics

M, B, H, and C are logical roles. Equality is permitted only for role pairs
whose Git semantics explicitly allow it; the initial supported equality is
`M == B`.

- `M == B` is valid and means that main has not drifted since divergence.
- An equal-revision comparison is a canonical, hash-bound empty comparison.
- Other equalities remain invalid in the initial OpenMRS/GitHub merge-preview
  boundary unless a later design establishes their exact coherent semantics.
- Candidate parentage, fetched refs, GitHub metadata, and freshness are still
  checked fail-closed.

Each comparison has one explicit result:

- `changed`;
- `unchanged`;
- `unsupported`;
- `analysis_limit_exceeded`;
- `acquisition_failed`.

No downstream stage may infer a missing comparison from an exception or empty
field.

## 5. Shared Model-Evidence Envelope

Risk, testability, and Gherkin generation must use the same evidence-boundary
abstraction rather than independent prompt payloads.

Each `ModelEvidenceEnvelope` records:

- snapshot key and comparison artifact hashes;
- model stage and reviewed-artifact input hashes;
- ordered model-visible anchors;
- the exact visible text and hash of every excerpt;
- selection score and reason for every visible anchor;
- an inventory and reason for omitted anchors;
- request byte and token budgets;
- output-schema hash;
- canonical envelope hash.

Local validators may accept a model citation only when it resolves to an
excerpt in that model call's envelope. The existence of additional hidden local
context is not sufficient.

All three live model stages must use one provider-size policy and must record
safe failure provenance. No stage may silently truncate until a request happens
to fit.

## 6. Bounded Frozen-Evidence Retrieval

When initial evidence is insufficient, TriageGuard uses a bounded retrieval
loop over the already frozen Git objects:

1. Deterministic ranking selects the initial evidence envelope.
2. The model may return a structured evidence need rather than inventing a
   conclusion.
3. Local code validates the need and searches only M, B, H, and C.
4. A bounded number of matching anchors is added to a successor context.
5. A new envelope and context hash are recorded.
6. Downstream model and review artifacts are invalidated.
7. The loop stops at configured round, anchor, byte, and time limits.

Exhaustion ends with an evidence-insufficient result. It never proves safety.

## 7. Validation Vocabulary

TriageGuard must not use one word, such as "grounded," for different strengths
of evidence.

### 7.1 Citation validated

Every cited identifier resolves to model-visible frozen evidence from the exact
request envelope.

### 7.2 Structurally testable

Approved evidence supplies identifiable setup, action, and observable roles
supported by the current execution adapter.

### 7.3 Human reviewed

A reviewer accepts or edits the unconfirmed risk and later accepts or edits the
scenario. Edits create new immutable artifacts and invalidate incompatible
downstream work.

### 7.4 Execution supported

The same validated test produces complete, attributable observations against B
and C, with controls and repetition sufficient for an approved classifier.

### 7.5 Severity eligible

A provisional CVSS assessment is allowed only when the classified runtime
transition and impact mapping meet the predefined eligibility policy.

None of the first three levels alone proves that a vulnerability exists.

## 8. Constrained Test and Execution Bridge

Milestone 2's approved Gherkin must become the only input to a generalized form
of Milestone 1's constrained execution machinery.

The bridge performs these steps:

1. Generate a plan using an allowlisted OpenMRS operation catalog.
2. Validate that every plan operation is supported by the reviewed risk,
   Gherkin, evidence envelope, and execution adapter.
3. Generate pytest without arbitrary imports, shell commands, database access,
   network destinations, or hidden executable content.
4. Validate the full source using AST, symbol, data-flow, and exact-step checks.
5. Build isolated targets for the exact B and C revisions.
6. Run the identical approved test and controls against both targets.
7. Repeat according to a recorded policy.
8. Persist raw observations before classification.
9. Classify only predefined complete behavioral pairs.

Unknown setup, build, execution, observation, or classification states remain
unknown. They are not converted into vulnerable, safe, or CVSS-bearing results.

## 9. Architecture Boundaries

New work must avoid further growth of the current workflow monolith. The target
boundaries are:

- `SnapshotService`: acquisition, role coherence, and freshness;
- `ComparisonBuilder`: changed and unchanged comparison artifacts;
- `EvidenceCatalog`: immutable full frozen evidence;
- `EvidenceEnvelopeBuilder`: stage-specific visible evidence and budgets;
- `ModelStageRunner`: common provider invocation and safe provenance;
- `RiskService`, `TestabilityService`, and `GherkinService`: stage contracts and
  deterministic validation;
- `ExecutionPlanService`: approved Gherkin to constrained plan;
- `ExecutionAdapter`: repository/system-specific environment and operations;
- `DifferentialRunner`: identical B/C execution and raw observations;
- `EvidenceClassifier`: predefined behavioral classification;
- `SeverityGate`: CVSS eligibility and calculation;
- `RunRepository`: immutable artifacts, journaling, recovery, and sealing;
- `WorkflowCoordinator`: state transitions only.

Refactoring is performed incrementally when a boundary is needed. A wholesale
rewrite is not part of this design.

## 10. Evaluation Design

### 10.1 Compatibility pilot

Begin with 25–30 stratified OpenMRS PRs covering Java-only, mixed artifact,
no-drift, drifted, rename/delete, binary, fork, large diff, parse failure,
conflict, and non-default-base cases.

Measure stage reachability, reason codes, evidence coverage, provider request
size, cost, latency, abstention, and human intervention.

### 10.2 Security evaluation cohorts

Use three distinct cohorts:

- reconstructed historical benign changes;
- reconstructed known security-relevant changes where evidence is available;
- controlled PR-shaped security mutations with executable ground truth.

Prospective open PRs measure operational compatibility, not accuracy, because
their true security outcome is generally unknown.

### 10.3 Baselines

- diff-only LLM;
- four-role/context LLM without TriageGuard gates;
- complete TriageGuard assurance chain.

### 10.4 Ablations

Remove one boundary at a time: base drift, merge preview, bounded retrieval,
testability gate, human review, Gherkin validation, constrained source
validation, or differential execution.

### 10.5 Primary metrics

- preparation and stage-completion rates;
- evidence-envelope coverage and provider reliability;
- hypothesis precision/recall where ground truth exists;
- citation validity and expert-rated semantic support;
- abstention appropriateness;
- testability and executable-test precision;
- generated-source approval and execution rates;
- behavioral-regression detection and false-positive rates;
- human edit frequency and review time;
- latency, token use, and cost;
- reproducibility across repeats, model versions, and recovery.

## 11. Ordered Delivery Gates

### Gate A — Correct and bounded front half

- logical snapshot roles permit legitimate equality;
- unchanged comparisons are first-class artifacts;
- all model stages use hashed visible evidence envelopes;
- every model request fits a declared budget before provider invocation;
- safe diagnostics and recovery are consistent across stages.

### Gate B — Scoped real-PR compatibility

- the compatibility pilot is recorded and repeatable;
- supported and unsupported PR classes are documented;
- failures are cohort-level measurements, not case-by-case surprises;
- GitHub CI verifies tests, lint, compilation, and replay artifacts.

### Gate C — First complete assurance chain

- one approved OpenMRS/Java risk class reaches constrained pytest;
- the exact test runs against frozen B and C targets;
- raw observations, controls, repeats, and classification are attributable;
- CVSS remains unavailable unless runtime evidence is eligible.

### Gate D — Research evaluation

- baselines, ablations, and evaluation cohorts are frozen;
- human labeling and disagreement procedures are documented;
- results distinguish compatibility, usefulness, and security accuracy;
- claims and limitations match measured evidence.

Breadth beyond OpenMRS Core begins only after Gate C. Publication claims require
Gate D.

## 12. Immediate Priority Order

1. Correct equal-role snapshot and empty-comparison semantics.
2. Design and implement the shared model-evidence envelope.
3. Apply the shared size/failure policy to all model stages.
4. Build the compatibility pilot and stage telemetry.
5. Extract the constrained execution bridge from Milestone 1.
6. Complete one B-versus-C executable OpenMRS/Java path.
7. Freeze and run the first baseline and ablation protocol.

## 13. Non-Goals Before the First Complete Chain

- arbitrary GitHub repositories;
- broad multi-language semantic analysis;
- replacing human security judgment;
- autonomous PR blocking or merging;
- deriving CVSS from model prose;
- accepting newer or human-supplied evidence after snapshot acquisition;
- optimizing for UI polish before pipeline measurements;
- claiming semantic proof from identifier or citation checks;
- treating provider success or abstention as vulnerability-detection accuracy.

## 14. Definition of Success

The deep vertical strategy succeeds when TriageGuard can take a supported
OpenMRS Core PR, preserve its exact PR-time counterfactual state, produce a
human-approved evidence-attributable hypothesis and scenario, generate and
validate a constrained test, execute that test reproducibly against B and C,
and issue either an evidence-supported behavioral result or an explicit
non-conclusive terminal outcome without inventing missing facts.

That operational success is necessary but not sufficient for publication. A
defensible contribution additionally requires the Gate D evaluation to show
which assurance boundaries improve reliability and at what compatibility,
human-effort, latency, and cost trade-offs.
