# Gate A Evidence Boundary Implementation Plan

> **For Codex:** Follow this plan task by task using test-driven development.
> Read the governing design and roadmap first. Do not skip red tests, do not
> create repository helper scripts, and do not push or merge without explicit
> user approval.

**Goal:** Make the OpenMRS Core front half logically correct, provider-bounded,
and honest about exactly which frozen evidence each model stage could see.

**Architecture:** Preserve `ContextBundle` as the complete bounded frozen
evidence catalog. Introduce a smaller immutable `ModelEvidenceEnvelope` for each
model call, selected deterministically from that catalog and measured against
the exact provider wire body. Risk, testability, and Gherkin generation consume
their own stage envelope; validators may resolve citations only within that
envelope. Keep orchestration in `MilestoneTwoWorkflow`, but extract evidence
selection, request sizing, and stage execution behind focused services.

**Tech stack:** Python 3.11, Pydantic 2, dataclasses, Groq structured outputs,
Git CLI, tree-sitter Java, pytest, Ruff, Streamlit.

**Governing documents:**

- `docs/architecture/2026-08-17-deep-vertical-assurance-chain-design.md`
- `docs/plans/2026-08-17-deep-vertical-roadmap.md`

## Global Constraints

- Preserve the current uncommitted provider diagnostics while replacing the
  risk-only emergency compaction with the shared design.
- Do not reset, overwrite, or reformat unrelated user changes.
- Permit only the initially approved revision equality: `M == B`.
- Do not silently slice anchor text. Select complete `ContextAnchor.text`
  values or record why an anchor was omitted.
- Measure the exact Groq request dictionary, including schema, before calling
  the provider.
- Never persist API keys, authorization headers, raw provider exceptions, or
  complete live request bodies in diagnostics.
- Keep replay deterministic and network-free.
- Schema changes must update durable loading, replay fixtures, and tamper tests
  in the same task.
- Every coherent implementation increment gets a local commit after its tests
  pass. Push only after the user reviews the completed gate.

## Progress

- [x] Task 1 — Permit only coherent snapshot-role equality
- [x] Task 2 — Represent unchanged comparisons as canonical artifacts
- [x] Task 3 — Centralize exact provider request sizing and policy
- [x] Task 4 — Add the immutable model-evidence envelope
- [ ] Task 5 — Migrate risk generation and validation to visible evidence
- [ ] Task 6 — Migrate testability and Gherkin to visible evidence
- [ ] Task 7 — Add bounded frozen-evidence refinement at model boundaries
- [ ] Task 8 — Unify model-stage execution, failure provenance, and recovery
- [ ] Task 9 — Update replay and UI behavior without weakening gates
- [ ] Task 10 — Complete Gate A verification and record measurements

Task 1 completed on 2026-08-17. Focused snapshot, domain, and Git regressions:
50 passed. Combined working-tree regression suite: 755 passed.

Task 2 completed on 2026-08-17. Focused diff, context, durability, and recovery
regressions at the clean Task 2 commit: 104 passed. Full clean-commit regression
suite: 753 passed. The preserved uncommitted diagnostics add eight tests, so the
combined working tree passed 761.

Task 3 completed on 2026-08-17. Strict configuration, exact Groq body
serialization, safe local rejection provenance, and gateway regressions: 70
passed. The exact Task 3 commit passed all 770 collected regressions; the
combined working tree, including seven preserved diagnostic tests, passed 777.
The 7,000-byte default is an initial conservative operational policy, not a
claim about Groq's universal platform limit.

Task 4 completed on 2026-08-17. Immutable-envelope, catalog-partition,
canonical-order, exact-text, whole-anchor selection, stage-policy, and exact
request-size regressions: 22 passed. The exact Task 4 commit passed all 792
collected regressions; the combined working tree, including seven preserved
diagnostic tests, passed all 799.

---

## Task 1 — Permit Only Coherent Snapshot-Role Equality

**Files:**

- Modify: `src/triageguard/domain/pr_analysis.py`
- Modify: `tests/domain/test_pr_analysis.py`
- Modify: `tests/analysis/test_snapshot.py`

### Step 1: Add failing role-coherence tests

Add a factory parameter to the existing snapshot test helper and cover the
allowed and forbidden equalities explicitly:

```python
def test_snapshot_allows_merge_base_to_equal_current_base() -> None:
    snapshot = _snapshot_values()
    snapshot["base_sha"] = snapshot["merge_base_sha"]
    snapshot["base_tree_sha"] = snapshot["merge_base_tree_sha"]

    result = PullRequestSnapshot.from_identity(**snapshot)

    assert result.merge_base_sha == result.base_sha


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("merge_base_sha", "head_sha"),
        ("merge_base_sha", "candidate_sha"),
        ("base_sha", "head_sha"),
        ("base_sha", "candidate_sha"),
        ("head_sha", "candidate_sha"),
    ],
)
def test_snapshot_rejects_unsupported_equal_revision_roles(left: str, right: str) -> None:
    values = _snapshot_values()
    values[right] = values[left]

    with pytest.raises(ValueError, match="revision roles"):
        PullRequestSnapshot.from_identity(**values)
```

Also test that `M == B` does not require equal tree hashes by accident: the
commit identity is what permits the equality, while normal Git acquisition is
responsible for supplying coherent tree objects.

### Step 2: Run the red tests

```bash
.venv/bin/python -m pytest \
  tests/domain/test_pr_analysis.py \
  tests/analysis/test_snapshot.py \
  -q --tb=short
```

Expected: the `M == B` case fails with the current “must be distinct” validator.

### Step 3: Replace cardinality validation with named role rules

In `PullRequestSnapshot.validate_snapshot_coherence`, replace the set-length
check with explicit forbidden pairs:

```python
forbidden_equalities = (
    ("merge_base", self.merge_base_sha, "head", self.head_sha),
    ("merge_base", self.merge_base_sha, "candidate", self.candidate_sha),
    ("base", self.base_sha, "head", self.head_sha),
    ("base", self.base_sha, "candidate", self.candidate_sha),
    ("head", self.head_sha, "candidate", self.candidate_sha),
)
for left_role, left_sha, right_role, right_sha in forbidden_equalities:
    if left_sha == right_sha:
        raise ValueError(
            f"unsupported equal revision roles: {left_role} and {right_role}"
        )
```

Do not add a general `allow_equal_revisions` flag. Equality belongs to named
role semantics, not caller discretion.

### Step 4: Run focused and regression tests

```bash
.venv/bin/python -m pytest \
  tests/domain/test_pr_analysis.py \
  tests/analysis/test_snapshot.py \
  tests/sources/test_git.py \
  -q --tb=short
```

Expected: all pass.

### Step 5: Commit

```bash
git add src/triageguard/domain/pr_analysis.py \
  tests/domain/test_pr_analysis.py \
  tests/analysis/test_snapshot.py
git commit -m "fix: allow coherent no-drift snapshots"
```

---

## Task 2 — Represent Unchanged Comparisons as Canonical Artifacts

**Files:**

- Modify: `src/triageguard/domain/pr_analysis.py`
- Modify: `src/triageguard/analysis/diffs.py`
- Modify: `src/triageguard/analysis/snapshot.py`
- Modify: `tests/analysis/test_diffs.py`
- Modify: `tests/analysis/test_snapshot.py`
- Modify: `tests/domain/test_pr_analysis.py`
- Modify: `fixtures/milestone_two/openmrs_shaped_pr/metadata/repository.json`
- Modify: affected replay/durability fixtures that serialize `DiffArtifact`

### Step 1: Add failing unchanged-comparison tests

The canonical no-drift artifact must be explicit and empty:

```python
def test_parse_patch_builds_canonical_unchanged_base_drift() -> None:
    sha = "a" * 40

    artifact = parse_patch(
        kind="base_drift_diff",
        old_revision=sha,
        new_revision=sha,
        patch_bytes=b"",
        numstat_bytes=b"",
        git_version="git version 2.50.1",
        max_files=1_000,
        max_bytes=16_000_000,
    )

    assert artifact.comparison_status == "unchanged"
    assert artifact.files == ()
    assert artifact.patch_sha256 == hashlib.sha256(b"").hexdigest()
```

Add rejection tests for:

- equal revisions with `author_diff`;
- equal revisions with `integration_diff`;
- equal revisions plus non-empty patch or numstat;
- non-empty content marked `unchanged` in persisted input;
- empty distinct-revision comparison marked `changed`.

### Step 2: Run the red tests

```bash
.venv/bin/python -m pytest \
  tests/analysis/test_diffs.py \
  tests/domain/test_pr_analysis.py \
  -q --tb=short
```

Expected: equal revisions are rejected and `comparison_status` is absent.

### Step 3: Add the explicit status to `DiffArtifact`

Use a required field, not a default that could reinterpret old artifacts:

```python
comparison_status: Literal["changed", "unchanged"]
```

Validate this invariant:

```python
empty_patch_sha256 = hashlib.sha256(b"").hexdigest()
same_revision = self.old_revision == self.new_revision
if self.comparison_status == "changed":
    if not self.files or self.patch_sha256 == empty_patch_sha256 or same_revision:
        raise ValueError("changed comparison requires distinct changed content")
elif self.files or self.patch_sha256 != empty_patch_sha256:
    raise ValueError("unchanged comparison requires canonical empty content")
if same_revision and self.kind != "base_drift_diff":
    raise ValueError("only base drift supports equal revisions")
```

Include `comparison_status` in canonical artifact hash input. Update every
fixture through its existing construction helper; do not add permissive load
migration that guesses status from missing data.

### Step 4: Teach `parse_patch` the single allowed empty case

Before normal file parsing:

```python
if old_revision == new_revision:
    if kind != "base_drift_diff" or patch_bytes or numstat_bytes:
        raise DiffBuildError(
            "diff_revision_mismatch",
            "Only an empty base-drift comparison may use one revision.",
        )
files = (
    ()
    if patch_bytes == b"" and numstat_bytes == b""
    else tuple(_parse_file_blocks(patch_bytes, _parse_numstat(numstat_bytes)))
)
comparison_status = "changed" if files else "unchanged"
```

Use the existing error constructor signature in `diffs.py`; preserve its public
reason-code behavior.

### Step 5: Verify acquisition and context behavior

Add an acquisition test proving `SnapshotAcquirer`/`DiffBuilder` returns three
artifacts when `M == B`, with `base_drift_diff.comparison_status ==
"unchanged"`. Add a context test proving an unchanged drift artifact produces
no drift anchors but does not make the primary integration evidence invalid.

```bash
.venv/bin/python -m pytest \
  tests/analysis/test_diffs.py \
  tests/analysis/test_snapshot.py \
  tests/analysis/test_context.py \
  tests/workflow/test_milestone_two_prepare.py \
  -q --tb=short
```

Expected: all pass.

### Step 6: Commit

```bash
git add src/triageguard/domain/pr_analysis.py \
  src/triageguard/analysis/diffs.py \
  src/triageguard/analysis/snapshot.py \
  tests fixtures/milestone_two/openmrs_shaped_pr
git commit -m "feat: record canonical unchanged base drift"
```

Before committing, inspect `git diff --cached --stat` and unstage unrelated
files if the broad `tests` path selected any.

---

## Task 3 — Centralize Exact Provider Request Sizing and Policy

**Files:**

- Create: `src/triageguard/llm/request_budget.py`
- Modify: `src/triageguard/llm/groq_gateway.py`
- Modify: `src/triageguard/llm/__init__.py`
- Modify: `src/triageguard/config.py`
- Modify: `tests/llm/test_request_budget.py`
- Modify: `tests/llm/test_gateways.py`
- Modify: `tests/test_config.py`

### Step 1: Add failing policy and serialization tests

Cover strict positive integers, environment loading, exact canonical body
measurement, and a local preflight rejection that makes no client call:

```python
def test_groq_body_size_matches_the_exact_gateway_call() -> None:
    request = _request()
    body = groq_request_body(request=request, model="openai/gpt-oss-120b")

    assert groq_request_body_bytes(request=request, model="openai/gpt-oss-120b") == len(
        canonical_json(body).encode("utf-8")
    )


def test_gateway_rejects_an_oversized_body_before_calling_groq() -> None:
    client = _RecordingClient()
    gateway = GroqStructuredGateway(
        _live_settings(max_model_request_bytes=400),
        client=client,
    )

    with pytest.raises(ModelRequestTooLarge, match="declared provider budget") as error:
        gateway.generate(_request_with_large_payload())

    assert client.calls == []
    assert error.value.provenance.reason_code == "model_request_too_large"
```

### Step 2: Run the red tests

```bash
.venv/bin/python -m pytest \
  tests/test_config.py \
  tests/llm/test_request_budget.py \
  tests/llm/test_gateways.py \
  -q --tb=short
```

Expected: missing policy types/settings and no local preflight guard.

### Step 3: Add strict settings and policy types

Add these settings with validated environment counterparts:

```python
max_model_request_bytes: int = 7_000
max_model_evidence_rounds: int = 2
```

The 7,000-byte value is a conservative initial operational policy derived from
the live 413 incident, not a claim about Groq's universal platform limit. Gate B
will measure whether to change it. Reject booleans, zero, and negative values.

Create:

```python
@dataclass(frozen=True)
class ProviderRequestBudget:
    provider: Literal["groq"]
    model: str
    max_body_bytes: int
    policy_version: Literal["groq-body-v1"] = "groq-body-v1"

    @classmethod
    def from_settings(cls, settings: Settings) -> ProviderRequestBudget: ...


class ModelRequestTooLarge(ModelGatewayError):
    """The exact provider body exceeds the locally declared safe budget."""
```

### Step 4: Extract one canonical Groq request-body builder

Move the `call_kwargs` construction from `GroqStructuredGateway.generate` into:

```python
def groq_request_body(*, request: ModelRequest, model: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": canonical_json(request.payload)},
        ],
        "model": model,
        "max_tokens": request.max_output_tokens,
        "reasoning_effort": "medium",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": request.purpose,
                "strict": True,
                "schema": request.output_schema,
            },
        },
    }
```

Both preflight and the client call must consume this exact dictionary. Delete
the risk-only duplicated wire-body builder only in Task 5, when its replacement
is in place.

### Step 5: Record safe local rejection provenance

Create one failed `ModelAttempt` with exact `request_body_bytes` and configured
`provider_body_limit_bytes`; set `status_code=None` because no provider call
occurred. Hash the request and a fixed local error description, not its body.

### Step 6: Run tests and commit

```bash
.venv/bin/python -m pytest \
  tests/test_config.py \
  tests/llm/test_request_budget.py \
  tests/llm/test_gateways.py \
  -q --tb=short
.venv/bin/python -m ruff check src/triageguard/llm src/triageguard/config.py tests/llm tests/test_config.py
```

Expected: all pass.

```bash
git add src/triageguard/llm src/triageguard/config.py \
  tests/llm tests/test_config.py
git commit -m "feat: enforce exact model request budgets"
```

---

## Task 4 — Add the Immutable Model-Evidence Envelope

**Files:**

- Create: `src/triageguard/evidence/__init__.py`
- Create: `src/triageguard/evidence/model_envelope.py`
- Create: `src/triageguard/evidence/selection.py`
- Create: `tests/evidence/test_model_envelope.py`
- Create: `tests/evidence/test_selection.py`
- Modify: `src/triageguard/domain/models.py` only if shared hash types must be
  exported

### Step 1: Add failing envelope integrity tests

Test immutable hashing, full catalog partitioning, exact visible text hashes,
ordering, omission reasons, stage-specific required anchors, and whole-anchor
selection:

```python
def test_envelope_hashes_exact_visible_text_and_partitions_catalog() -> None:
    context = _context_with_three_anchors()

    envelope = ModelEvidenceEnvelope.from_content(
        stage="risk_hypothesis",
        snapshot_key=context.snapshot_key,
        context_sha256=context.context_sha256,
        comparison_bindings=_comparison_bindings(),
        input_bindings=(),
        visible_anchors=(_visible(context.anchors[0]),),
        omitted_anchors=(
            OmittedEvidenceAnchor(anchor_id=context.anchors[1].anchor_id, reason="request_budget"),
            OmittedEvidenceAnchor(anchor_id=context.anchors[2].anchor_id, reason="request_budget"),
        ),
        max_request_body_bytes=7_000,
        selection_policy_version="risk-evidence-v1",
        output_schema_sha256="f" * 64,
    )

    assert envelope.visible_anchors[0].visible_text == context.anchors[0].text
    assert set(envelope.catalog_anchor_ids) == {
        anchor.anchor_id for anchor in context.anchors
    }
```

Tampering any visible text, source hash, omission reason, input binding, stage,
budget, schema hash, or envelope hash must fail validation.

### Step 2: Define strict envelope models

Implement these public shapes with `extra="forbid"` and frozen Pydantic
configuration:

```python
ModelEvidenceStage = Literal[
    "risk_hypothesis",
    "testability_assessment",
    "gherkin_generation",
]

class EvidenceArtifactBinding(ResearchArtifact):
    name: StrictStr = Field(min_length=1)
    sha256: Sha256

class VisibleEvidenceAnchor(ResearchArtifact):
    anchor_id: StrictStr = Field(min_length=1)
    revision_role: Literal["merge_base", "base", "head", "candidate"]
    path: StrictStr = Field(min_length=1)
    java_symbol: StrictStr | None
    start_line: StrictInt = Field(gt=0)
    end_line: StrictInt = Field(gt=0)
    change_relation: Literal[
        "author_change", "integration_change", "base_drift_change", "repository_context"
    ]
    visible_text: StrictStr = Field(min_length=1)
    source_text_sha256: Sha256
    visible_text_sha256: Sha256
    selection_reason: StrictStr = Field(min_length=1)

class OmittedEvidenceAnchor(ResearchArtifact):
    anchor_id: StrictStr = Field(min_length=1)
    reason: Literal["request_budget", "stage_irrelevant", "superseded"]

class ModelEvidenceEnvelope(ResearchArtifact):
    stage: ModelEvidenceStage
    snapshot_key: Sha256
    context_sha256: Sha256
    comparison_bindings: tuple[EvidenceArtifactBinding, ...]
    input_bindings: tuple[EvidenceArtifactBinding, ...]
    visible_anchors: tuple[VisibleEvidenceAnchor, ...]
    omitted_anchors: tuple[OmittedEvidenceAnchor, ...]
    catalog_anchor_ids: tuple[StrictStr, ...]
    max_request_body_bytes: StrictInt = Field(gt=0)
    selection_policy_version: StrictStr = Field(min_length=1)
    output_schema_sha256: Sha256
    envelope_sha256: Sha256
```

Require unique names and anchor IDs. Require visible and omitted IDs to be
disjoint and to partition `catalog_anchor_ids` exactly. `from_content` computes
the canonical hash.

### Step 3: Add a deterministic whole-anchor selector

Use a request factory and the exact body sizer so selection accounts for the
system prompt and output schema:

```python
@dataclass(frozen=True)
class EnvelopeBuildResult:
    envelope: ModelEvidenceEnvelope
    request: ModelRequest
    request_body_bytes: int


class EvidenceEnvelopeBuilder:
    def build(
        self,
        *,
        stage: ModelEvidenceStage,
        context: ContextBundle,
        comparison_bindings: tuple[EvidenceArtifactBinding, ...],
        input_bindings: tuple[EvidenceArtifactBinding, ...],
        required_anchor_ids: tuple[str, ...],
        priority_terms: tuple[str, ...],
        budget: ProviderRequestBudget,
        request_factory: Callable[[ModelEvidenceEnvelope], ModelRequest],
    ) -> EnvelopeBuildResult: ...
```

Algorithm:

1. Validate required IDs against the catalog.
2. Rank complete anchors deterministically by required status, exact priority
   term match, change relation, existing score components, path, start line,
   and anchor ID.
3. Add one complete anchor at a time.
4. Build a candidate envelope and complete `ModelRequest`.
5. Measure the exact provider body.
6. Keep the anchor only when the request remains within budget.
7. Fail with `ModelEvidenceBudgetError` if any required anchor cannot fit.
8. Record every unselected anchor and reason.

Do not slice `anchor.text`. Do not mutate `ContextBundle`.

### Step 4: Add stage-policy tests

Risk selection must reserve at least one integration anchor, then one author
anchor when present. It may select drift evidence only when the base-drift
comparison is `changed`. Testability requires the reviewed risk's cited
anchors. Gherkin requires the union of reviewed-risk citations and validated
setup/action/observable bindings.

### Step 5: Run tests and commit

```bash
.venv/bin/python -m pytest tests/evidence -q --tb=short
.venv/bin/python -m ruff check src/triageguard/evidence tests/evidence
```

Expected: all pass.

```bash
git add src/triageguard/evidence tests/evidence src/triageguard/domain/models.py
git commit -m "feat: bind model calls to exact visible evidence"
```

---

## Task 5 — Migrate Risk Generation and Validation to Visible Evidence

**Files:**

- Modify: `src/triageguard/hypotheses/generator.py`
- Modify: `src/triageguard/hypotheses/validator.py`
- Modify: `src/triageguard/domain/pr_analysis.py`
- Modify: `src/triageguard/workflow/milestone_two.py`
- Modify: `tests/hypotheses/test_generator.py`
- Modify: `tests/hypotheses/test_risk_validator.py`
- Modify: `tests/workflow/test_milestone_two_risks.py`
- Modify: `tests/workflow/test_milestone_two_durable_risk_replay.py`

### Step 1: Add failing visibility-boundary tests

Create a context with one visible and one omitted anchor. A draft that cites the
omitted anchor must fail even though that anchor exists in `ContextBundle`:

```python
def test_validator_rejects_citation_to_hidden_context_anchor() -> None:
    context = _context_with_visible_and_hidden_anchor()
    envelope = _risk_envelope(context, visible_ids=("anchor-visible",))
    draft = _risk_draft(citation_anchor_ids=("anchor-hidden",))

    assessment, report = validate_risk_assessment(
        draft=draft,
        snapshot=_snapshot(),
        context=context,
        evidence_envelope=envelope,
    )

    assert assessment is None
    assert "citation_not_model_visible" in report.reason_codes
```

Also test:

- identifiers are searched only in `VisibleEvidenceAnchor.visible_text`;
- the model must echo `evidence_envelope_sha256`;
- tampered envelope hashes fail before semantic validation;
- the exact built risk request remains within policy;
- the selector never returns a partial anchor string;
- the emergency 200-byte excerpt behavior no longer exists.

### Step 2: Extend risk artifacts with envelope identity

Add `evidence_envelope_sha256: Sha256` to `RiskAssessmentDraft`,
`RiskAssessment`, and `GroundingReport`. Include it in derived hashes and
durable validation.

Update the JSON schema so all outcomes echo the request envelope hash.

### Step 3: Replace the risk-only compactor

Change `build_risk_request` to require a prebuilt envelope and serialize only
its visible evidence:

```python
def build_risk_request(
    *,
    snapshot: PullRequestSnapshot,
    diffs: Sequence[DiffArtifact],
    context: ContextBundle,
    evidence_envelope: ModelEvidenceEnvelope,
) -> ModelRequest:
    _validate_frozen_inputs(snapshot, diffs, context)
    _validate_envelope_binding(
        stage="risk_hypothesis",
        snapshot=snapshot,
        context=context,
        envelope=evidence_envelope,
    )
    return ModelRequest(
        purpose="risk_hypothesis",
        system_prompt=RISK_SYSTEM_PROMPT,
        payload={
            "snapshot": _snapshot_payload(snapshot),
            "comparisons": _comparison_payloads(diffs),
            "evidence_envelope": evidence_envelope.model_dump(mode="json"),
            "output_rules": _risk_output_rules(),
        },
        output_schema=RISK_OUTPUT_SCHEMA,
        max_output_tokens=4096,
    )
```

Delete `_RISK_REQUEST_MAX_BODY_BYTES`, `_RISK_CONTEXT_EXCERPT_BYTES`, and the
risk-specific wire-body estimator after the shared builder is exercised by
tests.

### Step 4: Validate only model-visible evidence

Build `anchors_by_id` from `evidence_envelope.visible_anchors`, not
`context.anchors`. Validate envelope snapshot/context/stage/schema/budget
binding before checking any draft field.

Rename user-facing/internal claims from “grounded” to “citation validated” where
they currently imply semantic proof. Keep old public reason codes only where a
durable compatibility test requires them; otherwise introduce precise codes.

### Step 5: Persist the risk envelope before the model response

The workflow must write:

```text
artifacts/model_evidence/risk_hypothesis.json
artifacts/model_responses/risk_hypothesis.json
```

Hydration validates and restores the envelope before loading the response or
assessment. A missing, tampered, or mismatched envelope invalidates downstream
risk state.

### Step 6: Run tests and commit

```bash
.venv/bin/python -m pytest \
  tests/hypotheses/test_generator.py \
  tests/hypotheses/test_risk_validator.py \
  tests/workflow/test_milestone_two_risks.py \
  tests/workflow/test_milestone_two_durable_risk_replay.py \
  tests/workflow/test_milestone_two_tamper_recovery.py \
  -q --tb=short
```

Expected: all pass.

```bash
git add src/triageguard/hypotheses src/triageguard/domain/pr_analysis.py \
  src/triageguard/workflow/milestone_two.py tests/hypotheses tests/workflow
git commit -m "feat: validate risks against model-visible evidence"
```

Inspect the staged diff carefully because workflow tests contain unrelated
features.

---

## Task 6 — Migrate Testability and Gherkin to Visible Evidence

**Files:**

- Modify: `src/triageguard/testability/generator.py`
- Modify: `src/triageguard/testability/validator.py`
- Modify: `src/triageguard/contracts/gherkin_generation.py`
- Modify: `src/triageguard/domain/pr_analysis.py`
- Modify: `src/triageguard/workflow/milestone_two.py`
- Modify: `tests/testability/test_testability_generator.py`
- Modify: `tests/contracts/test_gherkin_generation.py`
- Modify: `tests/workflow/test_milestone_two_gherkin.py`
- Modify: `tests/workflow/test_milestone_two_durable_gherkin.py`

### Step 1: Add failing all-stage budget and visibility tests

Use a deliberately large context and assert that both stages produce bounded
requests containing only whole visible anchors:

```python
@pytest.mark.parametrize("stage", ["testability_assessment", "gherkin_generation"])
def test_stage_request_uses_only_its_bounded_visible_envelope(stage: str) -> None:
    result = _build_stage_with_large_context(stage)

    assert result.request_body_bytes <= result.envelope.max_request_body_bytes
    assert _payload_anchor_ids(result.request) == {
        anchor.anchor_id for anchor in result.envelope.visible_anchors
    }
```

Add hidden-citation rejection tests analogous to Task 5. Add a test proving
required risk/testability anchors cause a typed local budget failure rather than
being silently omitted.

### Step 2: Bind testability artifacts to their envelope

Add `evidence_envelope_sha256` to `TestabilityAssessmentDraft` and
`TestabilityAssessment`. Update schema, hashing, validator, persistence, and
recovery.

Required anchor IDs are the human-reviewed risk citations. Selection may add
other complete anchors after those fit.

### Step 3: Bind Gherkin artifacts to their envelope

Add `evidence_envelope_sha256` to `GherkinCandidateDraft` and
`GherkinCandidate`. Required anchor IDs are the union of:

- human-reviewed risk citations;
- locally validated testability setup bindings;
- locally validated action bindings;
- locally validated observable bindings.

`validate_gherkin_candidate` and edited-Gherkin validation resolve step evidence
only inside the Gherkin envelope.

### Step 4: Persist and hydrate both envelopes

Use stable paths:

```text
artifacts/model_evidence/testability_assessment.json
artifacts/model_evidence/gherkin_generation.json
```

Hydrate in dependency order: prepared context, risk envelope/assessment, human
review, testability envelope/assessment, Gherkin envelope/candidate, validated
edit, terminal record.

### Step 5: Run tests and commit

```bash
.venv/bin/python -m pytest \
  tests/testability/test_testability_generator.py \
  tests/contracts/test_gherkin_generation.py \
  tests/workflow/test_milestone_two_gherkin.py \
  tests/workflow/test_milestone_two_approval.py \
  tests/workflow/test_milestone_two_durable_gherkin.py \
  tests/workflow/test_milestone_two_tamper_recovery.py \
  -q --tb=short
```

Expected: all pass.

```bash
git add src/triageguard/testability \
  src/triageguard/contracts/gherkin_generation.py \
  src/triageguard/domain/pr_analysis.py \
  src/triageguard/workflow/milestone_two.py \
  tests/testability tests/contracts tests/workflow
git commit -m "feat: bind testability and gherkin to visible evidence"
```

---

## Task 7 — Add Bounded Frozen-Evidence Refinement at Model Boundaries

**Files:**

- Modify: `src/triageguard/domain/pr_analysis.py`
- Modify: `src/triageguard/analysis/refinement.py`
- Create: `src/triageguard/evidence/refinement.py`
- Modify: `src/triageguard/evidence/selection.py`
- Modify: `src/triageguard/workflow/milestone_two.py`
- Modify: `tests/analysis/test_refinement.py`
- Create: `tests/evidence/test_refinement.py`
- Modify: `tests/workflow/test_milestone_two_refinement.py`

### Step 1: Add failing bounded-loop tests

Cover these cases:

1. A model requests a precise identifier already present in an omitted catalog
   anchor; the next envelope prioritizes that anchor without changing the
   frozen context hash.
2. A requested identifier is absent from the catalog but present in an allowed
   M/B/H/C Java blob; the refiner creates a successor context and hash.
3. Search never reads a revision outside the snapshot.
4. A successor context invalidates risk, review, testability, and Gherkin state.
5. The configured round limit produces an explicit exhausted outcome.
6. A vague, empty, duplicate, or prohibited evidence need is rejected locally.

```python
def test_refinement_prioritizes_an_existing_omitted_anchor() -> None:
    result = resolver.resolve(
        snapshot=snapshot,
        context=context,
        needs=(_need(search_terms=("hasPrivilege",)),),
        completed_rounds=0,
        max_rounds=2,
    )

    assert result.context == context
    assert result.priority_anchor_ids == ("anchor-hidden-has-privilege",)
    assert result.exhausted is False
```

### Step 2: Use one structured evidence-need model

Risk insufficient-context output must use `FrozenEvidenceNeed`, not free-form
`needed_evidence` strings. Testability already uses this model. Every need has:

- a supported category;
- bounded exact search terms;
- a plain explanation;
- supporting model-visible anchor IDs.

Update the risk schema and fixture. Supporting citations still must be visible
in the envelope that requested refinement.

### Step 3: Generalize refinement input

Change `FrozenContextRefiner.refine` to accept validated
`Sequence[FrozenEvidenceNeed]` instead of the entire
`TestabilityAssessment`. Existing workflow callers pass
`assessment.evidence_needs`.

Add `FrozenEvidenceResolver`, which first searches the current catalog and then
delegates to `FrozenContextRefiner` only when required. Its result records:

```python
class EvidenceRefinementResult(ResearchArtifact):
    parent_context_sha256: Sha256
    successor_context_sha256: Sha256
    requested_need_sha256: Sha256
    priority_anchor_ids: tuple[StrictStr, ...]
    added_anchor_ids: tuple[StrictStr, ...]
    round_number: StrictInt = Field(gt=0)
    exhausted: StrictBool
    reason_code: StrictStr
    refinement_sha256: Sha256
```

### Step 4: Add workflow states without bypasses

Both a risk-level `insufficient_context_to_assess` and testability-level
`needs_more_frozen_evidence` may enter one refinement state. A successful
refinement returns to `PREPARED` and clears every downstream artifact. An
exhausted loop may seal only the evidence-insufficient terminal outcome.

Persist every refinement before state mutation. Hydration must replay the chain
and reject missing parents, non-monotonic rounds, or hash mismatches.

### Step 5: Run tests and commit

```bash
.venv/bin/python -m pytest \
  tests/analysis/test_refinement.py \
  tests/evidence/test_refinement.py \
  tests/workflow/test_milestone_two_refinement.py \
  tests/workflow/test_milestone_two_durability.py \
  tests/workflow/test_milestone_two_tamper_recovery.py \
  -q --tb=short
```

Expected: all pass.

```bash
git add src/triageguard/domain/pr_analysis.py \
  src/triageguard/analysis/refinement.py \
  src/triageguard/evidence \
  src/triageguard/workflow/milestone_two.py \
  tests/analysis tests/evidence tests/workflow
git commit -m "feat: refine only bounded frozen model evidence"
```

---

## Task 8 — Unify Model-Stage Execution, Failure Provenance, and Recovery

**Files:**

- Create: `src/triageguard/llm/stage_runner.py`
- Modify: `src/triageguard/llm/gateway.py`
- Modify: `src/triageguard/workflow/milestone_two.py`
- Modify: `src/triageguard/research/recorder.py` only if a generic artifact
  namespace requires it
- Create: `tests/llm/test_stage_runner.py`
- Modify: `tests/workflow/test_milestone_two_durable_risk_replay.py`
- Modify: `tests/workflow/test_milestone_two_durable_gherkin.py`
- Modify: `tests/workflow/test_milestone_two_tamper_recovery.py`

### Step 1: Add failing cross-stage failure tests

Parameterize risk, testability, and Gherkin. For each stage verify:

- the envelope is saved before the call;
- a provider failure is saved under the correct stage;
- request bytes and declared limit are present;
- secrets and raw error text are absent;
- retry/resume reuses a successful durable response;
- a failed call is retriable without accepting partial downstream state.

### Step 2: Introduce a narrow stage runner

```python
@dataclass(frozen=True)
class ModelStageResult:
    envelope: ModelEvidenceEnvelope
    request: ModelRequest
    response: ModelResponse


class ModelStageRunner:
    def run(
        self,
        *,
        run_handle: RunHandle,
        envelope: ModelEvidenceEnvelope,
        request: ModelRequest,
        gateway: StructuredModelGateway,
    ) -> ModelStageResult: ...
```

Responsibilities are limited to binding, exact size verification, persistence
ordering, invocation, and safe failure persistence. It does not interpret risk,
testability, or Gherkin content.

### Step 3: Replace risk-specific workflow failure paths

Use stage-keyed artifact paths and state:

```text
artifacts/model_evidence/{stage}.json
artifacts/model_responses/{stage}.json
artifacts/model_failures/{stage}.json
```

Delete `_record_risk_failure` after all callers use the runner. Expose a public
read-only `model_failure(stage)` accessor instead of more per-stage fields.

### Step 4: Verify recovery ordering

Hydration rejects a response without an envelope, an assessment without a
response, or any mismatched request/envelope/input hash. A sealed terminal
record remains immutable.

### Step 5: Run tests and commit

```bash
.venv/bin/python -m pytest \
  tests/llm/test_stage_runner.py \
  tests/workflow/test_milestone_two_durable_risk_replay.py \
  tests/workflow/test_milestone_two_durable_gherkin.py \
  tests/workflow/test_milestone_two_durability.py \
  tests/workflow/test_milestone_two_tamper_recovery.py \
  -q --tb=short
```

Expected: all pass.

```bash
git add src/triageguard/llm src/triageguard/workflow/milestone_two.py \
  src/triageguard/research/recorder.py tests/llm tests/workflow
git commit -m "refactor: unify durable model stage execution"
```

---

## Task 9 — Update Replay and UI Behavior Without Weakening Gates

**Files:**

- Modify: `src/triageguard/workflow/milestone_two_replay.py`
- Modify: `fixtures/milestone_two/openmrs_shaped_pr/model/risk_hypothesis.json`
- Modify: `fixtures/milestone_two/openmrs_shaped_pr/model/testability_assessment.json`
- Modify: `fixtures/milestone_two/openmrs_shaped_pr/model/gherkin_generation.json`
- Add or modify model-evidence fixture files under
  `fixtures/milestone_two/openmrs_shaped_pr/model/`
- Modify: `src/triageguard/ui/milestone_two.py`
- Modify: `src/triageguard/ui/milestone_two_presentation.py`
- Modify: `src/triageguard/ui/app.py`
- Modify: `tests/integration/test_milestone_two_replay.py`
- Modify: `tests/integration/test_milestone_two_outcomes.py`
- Modify: `tests/ui/test_milestone_two_ui.py`
- Modify: `tests/ui/test_milestone_two_presentation.py`

### Step 1: Add failing end-to-end presentation tests

The UI must show:

- each comparison on its own line with M/B/H/C plain-language explanations;
- shortened commit identifiers labeled as abbreviations, with full values in
  technical evidence;
- evidence coverage as “N of M frozen anchors visible to this model call”;
- explicit omitted-evidence reasons;
- stage-specific local-budget, provider, validation, and evidence-insufficient
  messages;
- a refinement action only when structured frozen evidence needs exist;
- no suggestion that citation validation proves a vulnerability.

### Step 2: Update replay fixtures to include envelopes

Replay must consume the same request shapes as live mode. Do not add replay-only
shortcuts that bypass envelope hash, size, citation, testability, or Gherkin
validation.

Add one no-drift replay case where `M == B` and the base-drift artifact is
`unchanged`.

### Step 3: Keep live errors safe and actionable

Map reason codes to stage-specific text. Example:

```python
if reason_code == "model_request_too_large":
    return (
        f"{stage_label} stopped before contacting {provider}: the exact "
        f"request was {request_bytes:,} bytes, above the declared "
        f"{limit_bytes:,}-byte policy. No conclusion was produced."
    )
```

Do not show keys, raw exceptions, response bodies, or prompts.

### Step 4: Run integration and UI tests

```bash
.venv/bin/python -m pytest \
  tests/integration/test_milestone_two_replay.py \
  tests/integration/test_milestone_two_outcomes.py \
  tests/ui/test_milestone_two_ui.py \
  tests/ui/test_milestone_two_presentation.py \
  tests/ui/test_app_smoke.py \
  -q --tb=short
```

Expected: all pass.

### Step 5: Commit

```bash
git add fixtures/milestone_two/openmrs_shaped_pr \
  src/triageguard/workflow/milestone_two_replay.py \
  src/triageguard/ui tests/integration tests/ui
git commit -m "feat: present bounded evidence outcomes end to end"
```

---

## Task 10 — Complete Gate A Verification and Record Measurements

**Files:**

- Create: `docs/evaluation/gate-a-front-half-report.md`
- Modify: `docs/plans/2026-08-17-deep-vertical-roadmap.md`
- Modify: this plan
- Modify: `README.md` only for supported behavior and run instructions

### Step 1: Add the Gate A report structure before measuring

Record:

- exact branch and commit;
- supported repository/language boundary;
- equality matrix;
- comparison status matrix;
- request policy version and configured bytes;
- per-stage exact request bytes for synthetic small, synthetic large, and one
  approved live OpenMRS run;
- visible/total anchor counts and omission reasons;
- refinement rounds and outcomes;
- replay/recovery results;
- known unsupported cases and remaining risks;
- the decision whether Gate B may begin.

Do not include API keys, live prompts, raw provider errors, or unpublished PR
security conclusions.

### Step 2: Run formatting, lint, compilation, and the full suite

```bash
.venv/bin/python -m ruff format --check \
  src/triageguard \
  tests \
  docs
.venv/bin/python -m ruff check src tests
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m pytest -q
git diff --check
```

If the existing repository-wide format baseline still contains unrelated old
files, run the check over every Gate A changed Python file and record the exact
pre-existing exception in the report. Do not reformat unrelated modules merely
to make the global check green.

### Step 3: Run the offline no-drift and large-context acceptance cases

```bash
.venv/bin/python -m pytest \
  tests/integration/test_milestone_two_replay.py \
  tests/integration/test_milestone_two_outcomes.py \
  -q --tb=short
```

Expected: deterministic pass with no network.

### Step 4: Run one user-approved live OpenMRS observation

Use the UI or a non-interactive diagnostic entry point with the user's existing
local credentials. Record only safe aggregate stage measurements. The live run
passes Gate A when every attempted request is locally within policy and every
outcome is typed and durable; the model does not have to propose a risk.

### Step 5: Review for architectural drift

Confirm:

- no stage serializes `ContextBundle.anchors` directly into a model request;
- no validator resolves citations from hidden catalog anchors;
- no risk-only request budget constant remains;
- no equal-revision exception is caller-configurable;
- no disposable `apply_*.py`, `fix_*.py`, or `add_*.py` files exist;
- workflow growth is limited to coordination and state transitions;
- every new durable artifact has tamper and recovery coverage.

### Step 6: Mark Gate A complete and commit documentation

Update all task checkboxes, the roadmap current gate, the decision log, and the
Gate A report with exact evidence.

```bash
git add README.md docs
git commit -m "docs: record gate a front half evidence"
```

Do not mark Gate A complete if any exit condition is unverified. Do not start
the Gate B detailed plan until the report is reviewed with the user.

## Gate A Completion Checklist

- [ ] `M == B` is accepted and all unsupported role equalities fail closed.
- [ ] Equal M/B produces a canonical unchanged base-drift artifact.
- [ ] All model stages use immutable hashed evidence envelopes.
- [ ] All citations resolve only to exact model-visible text.
- [ ] All exact provider bodies fit declared budgets before invocation.
- [ ] Required evidence that cannot fit causes a typed local failure.
- [ ] Frozen evidence refinement is structured, bounded, durable, and
  invalidates downstream state.
- [ ] Risk, testability, and Gherkin failures share safe durable provenance.
- [ ] Replay and recovery enforce the same boundary as live mode.
- [ ] UI language distinguishes citation validation, testability, human review,
  and execution support.
- [ ] Full verification and Gate A acceptance cases pass.
- [ ] Gate A report is reviewed before Gate B planning begins.

## Execution Handoff

When implementation begins, start with Task 1 and work inline in the current
task unless the user explicitly requests a separate task or subagent-driven
execution. At every commit boundary, report the exact tests run and remaining
unchecked tasks.
