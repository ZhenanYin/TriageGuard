# Gate A Front-Half Evidence Report

**Report status:** Work-in-progress review checkpoint — deterministic
measurements, repaired live OpenMRS observation, recovery verification, and
repository checks recorded; live request compatibility remains unresolved.

**Measurement date:** 2026-08-17

**Branch:** `feature/milestone-2`

**Measurement base commit:** `3198b7ac11c6bed8f68e424cda2b7c75fe5b7a66`

**Gate A documentation commit:** Pending

## 1. Decision Boundary

Gate A evaluates whether TriageGuard's OpenMRS Core front half is correct,
bounded, attributable, recoverable, and honest about uncertainty. It does not
evaluate vulnerability-detection accuracy and does not prove that any pull
request is vulnerable or safe.

**Current decision:** The boundedness, failure-safety, durability, and recovery
conditions are supported by the recorded evidence. Gate A remains open because
the current 7,000-byte policy prevents PR 6456 from reaching the first model
stage. This checkpoint is intended for collaborator review before choosing the
compatibility repair.

## 2. Supported Boundary

The initial supported live boundary is deliberately narrow:

- repository: `openmrs/openmrs-core`;
- pull requests: open GitHub pull requests targeting the repository's default
  branch with reproducible head and merge-preview Git objects;
- source emphasis: Java-centered authorization and security behavior;
- model provider: Groq through strict structured-output requests;
- evidence: immutable excerpts selected only from frozen M, B, H, and C Git
  objects;
- terminal front-half outputs: typed abstention or human-approved,
  evidence-bound Gherkin.

Arbitrary repositories, arbitrary languages, human-supplied test facts,
runtime vulnerability proof, pytest generation, differential B/C execution,
and CVSS eligibility are outside Gate A.

## 3. Snapshot Equality Matrix

| Role pair | Allowed equal? | Meaning or rejection |
|---|---:|---|
| M and B | Yes | Main has not changed since the pull request diverged. |
| M and H | No | The proposed head must differ from the shared starting point. |
| M and C | No | The merge preview must not collapse to the shared starting point. |
| B and H | No | The pull-request head and current main remain distinct roles. |
| B and C | No | The merge preview must differ from current main. |
| H and C | No | The merge preview and head remain distinct Git commits. |

Equality is a named snapshot rule. Callers cannot enable arbitrary
equal-revision comparisons.

## 4. Comparison Status Matrix

| Comparison | Revisions | Supported artifact status | Meaning |
|---|---|---|---|
| Author change | M→H | `changed` | Exact pull-request author changes. |
| Merge impact | B→C | `changed` | Exact change that merging would apply to current main. |
| Main-branch drift | M→B | `changed` or `unchanged` | Changes to main while the pull request waited, or canonical empty content when M equals B. |

Acquisition, parsing, unsupported content, and configured-limit failures are
typed stage outcomes rather than absent comparison artifacts.

## 5. Request Policy

| Setting | Recorded value |
|---|---|
| Provider | `groq` |
| Model | `openai/gpt-oss-120b` |
| Declared maximum request body | 7,000 bytes |
| Risk selection policy | `risk-evidence-v1` |
| Testability selection policy | `testability-evidence-v1` |
| Gherkin selection policy | `gherkin-evidence-v1` |

The declared byte limit is an application policy established from the observed
provider incident. It is not represented as Groq's universal platform limit.
Every limit check uses the exact serialized provider request dictionary,
including its structured-output schema.

## 6. Deterministic Synthetic Measurements

### 6.1 Small OpenMRS-shaped replay

| Stage | Exact body bytes | Declared limit | Visible/total anchors | Omission reasons | Outcome |
|---|---:|---:|---:|---|---|
| Risk hypothesis | 6,978 | 7,000 | 2/2 | None | Within policy |
| Testability assessment | 6,921 | 7,000 | 1/2 | `request_budget` (1) | Within policy |
| Gherkin generation | 6,996 | 7,000 | 1/2 | `request_budget` (1) | Within policy |

### 6.2 Large frozen context

| Stage | Exact body bytes | Declared limit | Visible/total anchors | Omission reasons | Outcome |
|---|---:|---:|---:|---|---|
| Risk hypothesis | 7,130 attempted | 7,000 | No envelope/6 catalog anchors | `request_budget`; mandatory evidence could not fit | Typed local `model_request_too_large` stop |
| Testability assessment | 6,210 | 7,000 | 1/2 | `request_budget` (1) | Within policy |
| Gherkin generation | 6,450 | 7,000 | 1/2 | `request_budget` (1) | Within policy |

Large-context acceptance requires whole-anchor selection within policy or a
typed local stop. Silent text slicing is prohibited.

These are deterministic stage-local fixtures, not claims about provider-wide
capacity. The large risk catalog contained six anchors and 3,589 bytes of exact
text. The large testability and Gherkin catalogs each contained two anchors and
10,861 and 10,267 bytes respectively. The latter stages retained their one
mandatory whole anchor and recorded the optional anchor as omitted. The risk
stage could not retain both mandatory comparison anchors once the complete
catalog inventory was represented, so it produced no envelope and made no
provider call.

The small replay's Gherkin request has only four bytes of policy headroom. Gate
B must therefore measure a representative OpenMRS cohort before changing or
claiming adequacy of the current policy.

## 7. No-Drift Acceptance

| Check | Result |
|---|---|
| Snapshot relation | M equals B |
| M→B revisions | Old revision equals new revision |
| Comparison status | `unchanged` |
| Patch and file inventory | Empty patch SHA-256 `e3b0c442…b855`; zero files |
| Risk request envelope binding | Assessment hash matches the exact risk envelope |

## 8. Frozen-Evidence Refinement

| Property | Result |
|---|---|
| Structured evidence needs | Pass — closed categories and exact identifier terms are validated locally |
| Maximum rounds and bounds | Two rounds; 40 files, 80 anchors, 160,000 context bytes, and 7,000 bytes per model request by default |
| Successor context/envelope hashes | Pass — refinement and successor context hashes are revalidated during recovery |
| Downstream invalidation | Pass — incompatible risk, review, testability, and Gherkin state is cleared |
| Exhaustion outcome | Pass — terminal `insufficient_context_to_assess`, never a safety conclusion |

## 9. Replay, Recovery, and Tamper Results

| Check | Result |
|---|---|
| Approved replay reaches validated Gherkin | Pass |
| Non-risk and evidence-insufficient outcomes are terminal | Pass |
| Model-stage responses recover without duplicate calls | Pass |
| Tampered envelopes or artifacts fail closed | Pass |
| Full repository suite | 866 passed |

## 10. Live OpenMRS Observation

**Status:** Repaired observation completed for public PR
`openmrs/openmrs-core#6456`. The run stopped locally at the risk boundary,
persisted the stop, and did not contact Groq.

Only these safe aggregate fields may be recorded:

- public PR number or an explicitly approved anonymized identifier;
- terminal stage and typed outcome/reason code;
- exact request-body bytes and declared limit for each attempted model stage;
- visible/total anchor counts and closed omission reasons;
- refinement round count;
- attempt count, aggregate latency, and token counts when available.

Do not record API keys, authorization headers, prompts, raw model responses,
raw provider errors, or unpublished security conclusions.

| Stage | Exact body bytes | Declared limit | Visible/total anchors | Outcome/reason | Aggregate latency |
|---|---:|---:|---:|---|---:|
| Risk hypothesis | 12,589 attempted | 7,000 | No envelope/54 catalog anchors | `model_request_too_large`; local pre-provider stop | No provider call |
| Testability assessment | Not reached | 7,000 | Not reached | Not reached | Not applicable |
| Gherkin generation | Not reached | 7,000 | Not reached | Not reached | Not applicable |

The prepared snapshot and 54-anchor frozen context were durably recorded. No
risk evidence envelope, model response, provider-failure artifact, or provider
attempt event exists, which confirms that the policy stopped the request before
Groq. The first run exposed that the pre-envelope stop itself was not persisted.
A test-driven repair now records only the stage, prepared hashes, closed reason
code, exact attempted/limit bytes, catalog count, and timestamp. Recovery
revalidates the recorder journal and exact deterministic selection result, and
the UI reports zero attempts and zero provider latency. The repaired live run
created that exact nine-field artifact. Local recovery restored it, re-raised
the same 12,589-byte outcome, and left the event journal unchanged, confirming
that no new model activity occurred.

## 11. Architectural-Drift Audit

| Prohibition | Result | Evidence |
|---|---|---|
| No model stage serializes `ContextBundle.anchors` directly | Pass | Risk, testability, and Gherkin request builders accept `ModelEvidenceEnvelope`; raw context access is confined to selection and deterministic input checks. |
| No validator resolves model-output citations from hidden catalog anchors | Pass | Risk, testability, and Gherkin output validation resolve citations from `evidence_envelope.visible_anchors`; hidden-anchor regressions are explicit tests. |
| No risk-only request budget remains | Pass | One `DEFAULT_MAX_MODEL_REQUEST_BYTES` setting and `ProviderRequestBudget.from_settings` serve every model stage and the gateway. |
| No caller-configurable equal-revision exception exists | Pass | Snapshot roles permit only named M==B equality; `parse_patch` permits equal revisions only for `base_drift_diff`. |
| No disposable repository patch scripts exist | Pass | Repository search found no `apply_*.py`, `fix_*.py`, or `add_*.py` files. |
| Workflow remains coordination/state-transition focused | Pass with tracked debt | Gate A extracted envelope selection and stage execution; the pre-existing coordinator remains large and should continue shrinking incrementally. |
| New durable artifacts have tamper and recovery coverage | Pass | Focused envelope, refinement, stage-runner, tamper, and restart suite: 74 passed. |

## 12. Known Unsupported Cases and Remaining Risks

- Non-OpenMRS repositories and non-Java-centered changes are outside the
  supported live boundary.
- A valid bounded abstention does not establish safety.
- Citation validation proves only that a cited reference was visible and
  immutable; it does not establish semantic truth or vulnerability existence.
- Structural testability and approved Gherkin are not runtime evidence.
- Provider behavior, model quality, and repository buildability remain Gate B
  and Gate C concerns.
- Prospective open pull requests generally lack security ground truth and
  measure compatibility rather than detection accuracy.
- Risk-stage pre-envelope request-budget stops are now typed, aggregate-only,
  durable, and recoverable. The first live observation discovered this gap;
  the repaired live rerun and local recovery now pass.
- Public PR 6456 cannot currently reach hypothesis generation: its minimum
  mandatory risk request is 12,589 bytes under the current schema and evidence
  inventory, while policy permits 7,000 bytes. Durability fixes the integrity
  of this abstention, not the compatibility limitation itself.
- The next compatibility decision must be evidence-based: calibrate provider
  acceptance with safe synthetic probes and an OpenMRS cohort, reduce fixed
  request overhead without dropping mandatory whole evidence, or select a
  provider/model boundary that can accept the required request. Raising the
  limit from one PR observation alone is not justified.

## 13. Verification Log

| Command or check | Result |
|---|---|
| Repository-wide Ruff format check | Baseline exception: 27 pre-existing files would be reformatted; no unrelated mass-formatting performed |
| Gate A changed-file Ruff format check | 58 Python files already formatted |
| Ruff lint | Pass — `All checks passed!` |
| Python compilation | Pass |
| Full pytest suite before live repair | 866 passed in 36.71 seconds |
| Full pytest suite after durable preflight-stop repair | 874 passed in 36.43 seconds |
| Offline replay/no-drift acceptance | 7 targeted acceptance tests passed |
| `git diff --check` | Pass |

## 14. Gate A Exit Decision

**Decision:** Gate A remains open; publish this checkpoint for collaborator
review.

The repaired live observation recorded a bounded typed outcome durably, the
saved outcome recovered without provider activity, and offline/full-suite
verification passed. However, the supported live front half still cannot reach
hypothesis generation for PR 6456 under the declared policy. Commit and push
this report and repair as a reviewable feature-branch checkpoint; do not mark
Gate A complete or merge to `main` until the compatibility decision is reviewed.
