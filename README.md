# TriageGuard

TriageGuard is a research project whose long-term goal is to move OpenMRS Core
change review from passive vulnerability response toward proactive, executable
security evidence. The intended workflow is:

```text
PR diff + whole-system context
  -> grounded candidate risks
  -> human selection and editing
  -> Gherkin security scenario
  -> constrained generated pytest
  -> base/candidate execution
  -> evidence-gated provisional CVSS
  -> future GitHub check
```

The research question is whether a human-reviewed security-risk hypothesis can
be turned into a trustworthy executable experiment whose conclusions remain
limited to what the resulting evidence supports.

## What is implemented now

TriageGuard currently contains two separate research milestones.

**Milestone 1** is a controlled local vertical slice for an OpenMRS-shaped
patient-deletion authorization regression. It can generate constrained pytest,
run fixture comparisons, and calculate a provisional CVSS result only for that
controlled fixture. It is not evidence about a real OpenMRS pull request.

**Milestone 2** is a human-guided, evidence-bounded review of one OpenMRS Core
pull request. It freezes four exact code photographs before any model request:

- **M — shared starting point:** where the PR and main branch last matched.
- **B — current main:** what main looked like when analysis began.
- **H — PR head:** the author’s proposed code.
- **C — merge preview:** what main would look like if the PR merged then.

It compares author change (**M → H**), merge impact (**B → C**), and
main-branch drift (**M → B**). The model may propose readable, unconfirmed
risk hypotheses, testability assessments, and Gherkin candidates. Local,
deterministic validation decides whether each item is grounded in the frozen
code evidence.

Milestone 2 has five human-gated pages:

| Page | User action | System action |
|---|---|---|
| 1. Choose a pull request | Submit one OpenMRS Core PR URL | Freeze M/B/H/C, the three comparisons, and bounded Java evidence |
| 2. Understand the change | Review the saved photographs and comparisons | Request and locally ground possible risk hypotheses |
| 3. Review possible risks | Read and select one unconfirmed hypothesis | Preserve its cited frozen evidence |
| 4. Choose and edit one risk | Edit one readable paragraph and review testability | Require saved code evidence for setup, action, and observable outcome; refine only M/B/H/C when necessary |
| 5. Create and approve the scenario | Review or edit one Gherkin scenario | Validate it against the reviewed risk and current frozen evidence before approval |

Milestone 2 stops after an approved evidence-bound Gherkin scenario, a supported
non-risk outcome, or an explicit insufficient-frozen-evidence outcome. It does
not generate pytest, execute OpenMRS, calculate CVSS, or claim that a real PR
is safe.

## Model boundary and deterministic gates

Milestone 1 uses an LLM for exactly two bounded roles:

1. Produce a structured test plan from the approved risk contract and exact
   Gherkin.
2. Render that approved plan as constrained `pytest-bdd` source.

Replay mode supplies checked-in responses for both calls. Live mode can make
the same structured calls through Groq. Neither mode lets model output approve
the security meaning or establish a result.

The Gherkin renderer and semantic-alignment check, primitive and data-flow
validation, full-source AST and allowlist checks, isolated pytest execution,
raw-observation validation, authorized controls, repeat-stability analysis,
fixed differential classifier, CVSS eligibility decision and calculation,
hashing, and artifact recording are deterministic. The generated source cannot
introduce arbitrary HTTP or process helpers, skip the test, swallow failures,
weaken the two security oracles, change the actor or action, or invent runtime
evidence.

Failures and abstentions are explicit terminal research results:

- Missing replay data, malformed model output, or a provider failure yields
  `generation_abstained`; no fallback plan or test is substituted.
- Source that fails deterministic validation yields `validation_failed` and is
  not executed.
- Setup, timeout, malformed observation, missing control, or incomplete
  execution yields `execution_inconclusive`.
- Security-relevant facts that differ across repetitions yield
  `unstable_result`.
- Only complete, controlled, stable paired facts can yield a regression, fix,
  pre-existing-risk, or no-regression classification.

Abstained, failed, inconclusive, and unstable runs receive no vulnerability
conclusion and no CVSS score.

## Evidence and provisional CVSS 4.0

The experiment directly measures request outcomes, patient state after the
attempt, setup/action completion, the authorized deletion control, and
repeat-to-repeat stability. For the controlled regression, the secure base is
shown as **Not scored** rather than `0.0`, because the tested vulnerability was
not observed. The vulnerable candidate is eligible for the expert-authored
provisional profile and is shown as **7.1 High**:

```text
CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:N/VC:N/VI:H/VA:N/SC:N/SI:N/SA:N
```

The LLM neither selects these metrics nor calculates the score. The installed
`cvss>=3.6,<4` library calculates the vector deterministically after the
evidence gate. Each of the eleven Base metrics has an explicit rationale and a
source category:

| Metric | Value | Rationale | Evidence type |
|---|---:|---|---|
| Attack Vector (`AV`) | Network (`N`) | The target OpenMRS REST API is assumed to be remotely reachable in the intended deployment; the local loopback fixture does not prove this. | Deployment assumption |
| Attack Complexity (`AC`) | Low (`L`) | Exploitation uses one ordinary authenticated deletion request with no evasion condition. | Runtime design |
| Attack Requirements (`AT`) | None (`N`) | No specific deployment or execution precondition is assumed beyond the vulnerable authorization behavior. | Deployment assumption |
| Privileges Required (`PR`) | Low (`L`) | The actor is an authenticated clerk with `View Patients` but without `Delete Patients`. | Approved contract |
| User Interaction (`UI`) | None (`N`) | No person other than the attacking clerk participates in the tested action. | Runtime design |
| Vulnerable-system Confidentiality (`VC`) | None (`N`) | The scoped deletion behavior demonstrates no additional disclosure. | Standard interpretation of the observed scope |
| Vulnerable-system Integrity (`VI`) | High (`H`) | Unauthorized deletion of a patient record is judged to be an integrity violation with a potentially direct, serious healthcare consequence. | Expert judgment |
| Vulnerable-system Availability (`VA`) | None (`N`) | CVSS Availability concerns loss of service or resource availability at system level; this scoped data deletion is assessed as integrity impact. | Standard interpretation |
| Subsequent-system Confidentiality (`SC`) | None (`N`) | No confidentiality impact to a subsequent system is demonstrated. | Runtime design and bounded evidence scope |
| Subsequent-system Integrity (`SI`) | None (`N`) | No integrity impact to a subsequent system is demonstrated. | Runtime design and bounded evidence scope |
| Subsequent-system Availability (`SA`) | None (`N`) | No availability impact to a subsequent system is demonstrated. | Runtime design and bounded evidence scope |

These categories keep three kinds of claim separate: measured runtime facts,
deployment assumptions such as remote reachability, and expert judgment such as
the seriousness of patient-record integrity loss. Pytest measures only the first
kind. A reviewer can challenge the assumptions and judgment independently. No
score delta is calculated, and a secure, unstable, inconclusive, or otherwise
ineligible side remains **Not scored**.

## Research contribution and evaluation plan

The working contribution hypothesis is that TriageGuard can produce trustworthy
executable security evidence from a human-reviewed risk hypothesis by binding
the approved meaning to constrained generation, paired execution, explicit
provenance, and evidence-gated severity. "Trustworthy" here means auditable and
fail-closed within the tested scope; it does not mean that the prototype has
already found real risks or has been validated at publication scale.

A future OpenMRS evaluation should combine a corpus of real pull requests with
seeded or mutation-based authorization regressions. Independent experts should
label risk hypotheses, Gherkin scenarios, generated tests, and relevant
base-versus-PR outcomes. The study should report:

- pass, rejection, abstention, inconclusive, and instability rates at every
  pipeline stage;
- plan, Gherkin, generated-test, oracle, execution, classification, and CVSS
  correctness against expert labels;
- detection performance on natural and seeded/mutated cases, including
  precision, recall, and false-positive/false-negative behavior where defined;
- repeat stability and sensitivity to repeated model and execution runs;
- provider and execution cost, token use, and end-to-end latency;
- expert review/editing time and total human effort;
- inter-rater and system-expert agreement;
- confidence intervals, plus statistical tests and effect sizes justified by
  the comparison design after sample-size and power planning.

Milestone 1 records the raw material for descriptive workflow measurements, but
this repository intentionally reports no invented evaluation numbers.

## Current limitations and non-claims

- There is no real OpenMRS diff/context adapter or system-aware risk proposal
  stage yet; the single risk contract is prepared and fixture-specific.
- The local fixture is OpenMRS-shaped but is not OpenMRS and does not validate
  real checkout, build, migration, configuration, or deployment behavior.
- Base/PR execution and a GitHub check are future work. This repository does not
  claim GitHub Actions integration or continuous pull-request monitoring.
- The prototype does not claim to discover production vulnerabilities, measure
  general risk-recall performance, or cover security categories beyond the
  controlled patient-deletion authorization case.
- The `7.1 High` assessment is provisional and expert-authored. Runtime evidence
  gates its use but does not empirically establish every metric.
- No expert study, baseline comparison, power analysis, confidence interval,
  hypothesis test, effect size, or publication-scale validation has been
  completed.
- Live model behavior can vary and incur provider cost. Deterministic rejection
  and abstention contain unsupported output but do not guarantee useful output.

## Repository structure

The Milestone 1 application is at the repository root:

```text
TriageGuard/
├── fixtures/patient_delete_authorization/  Controlled contract, model replies, and CVSS profile
├── src/triageguard/                         Application, workflow, gates, execution, UI, and recording
├── tests/                                   Offline unit, integration, durability, and UI tests
├── .env.example                             Non-secret environment-variable template
├── pyproject.toml                           Package metadata and development dependencies
└── README.md                                Research scope and operating guide
```

## Local setup

Run every command from the repository root with Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

### Milestone 2 replay setup (recommended first run)

The Milestone 2 replay interface uses an OpenMRS-shaped synthetic fixture and
checked-in model responses. It makes no GitHub or provider request and requires
no API key. It is useful for reviewing the interface and workflow, but it is
not a real pull-request analysis.

```bash
source .venv/bin/activate
export TRIAGEGUARD_LLM_MODE=replay
export TRIAGEGUARD_ENVIRONMENT_KIND=controlled_fixture
python -m streamlit run src/triageguard/ui/app.py
```

Open the local URL printed by Streamlit. The default URL is a reserved synthetic
OpenMRS-shaped PR. Follow the five pages to inspect saved comparisons, read and
edit one unconfirmed risk paragraph, check testability, and approve a Gherkin
scenario.

### Milestone 2 live OpenMRS Core mode

Live mode reads one public, open OpenMRS Core PR from GitHub, freezes its exact
M/B/H/C evidence, and asks Groq for structured risk, testability, and Gherkin
proposals. Keep credentials in the local process environment only; never paste
them into chat, source, fixtures, artifacts, or commits. A GitHub token is
recommended for API rate limits but is not required for a public repository.

Run this from the repository root. The two `read -s` commands prompt without
echoing a secret or placing it in shell history:

```bash
source .venv/bin/activate
export TRIAGEGUARD_LLM_MODE=live
export TRIAGEGUARD_LLM_PROVIDER=groq
export TRIAGEGUARD_LLM_MODEL=openai/gpt-oss-120b
export TRIAGEGUARD_ENVIRONMENT_KIND=real_pr_analysis
export TRIAGEGUARD_ARTIFACTS_DIR=artifacts/milestone-two-live
export TRIAGEGUARD_ANALYSIS_CACHE_DIR=analysis-cache/milestone-two-live
read -s "GROQ_API_KEY?Groq API key: "
export GROQ_API_KEY
read -s "GITHUB_TOKEN?GitHub token (recommended; Enter to skip): "
export GITHUB_TOKEN
test -n "$GROQ_API_KEY"
python -m streamlit run src/triageguard/ui/app.py
```

Before submitting a PR URL, confirm the UI reports **Live · groq** and the
intended model. The first live run should use a public, open OpenMRS Core PR and
should be reviewed as an engineering smoke test, not as a vulnerability finding.
There is no replay fallback: a missing key, unsupported PR, GitHub error, or
provider failure stops the run without inventing a result.

After stopping Streamlit with <kbd>Ctrl</kbd>+<kbd>C</kbd>, clear the session in
that same shell:

```bash
unset GROQ_API_KEY GITHUB_TOKEN TRIAGEGUARD_LLM_MODE TRIAGEGUARD_LLM_PROVIDER
unset TRIAGEGUARD_LLM_MODEL TRIAGEGUARD_ARTIFACTS_DIR
unset TRIAGEGUARD_ANALYSIS_CACHE_DIR TRIAGEGUARD_ENVIRONMENT_KIND
deactivate
```

## Tests and static checks

The checks use prerecorded responses, fake HTTP sessions, and local loopback
fixture servers; they need no API key or external network access:

```bash
source .venv/bin/activate
python -m pytest -q
python -m ruff check src tests
```

## Research artifacts and cleanup

Each run is stored beneath `TRIAGEGUARD_ARTIFACTS_DIR` in its own
`run-<id>/` directory. With the replay command above, the layout is:

```text
artifacts/run-<id>/
├── .run-ownership.json
├── events.jsonl
├── run_record.json
├── artifacts/
│   ├── prepared/<sha256>.json
│   ├── approved/<sha256>.json
│   ├── generated/<sha256>.json
│   ├── validated/<sha256>.json
│   ├── executed/<sha256>.json
│   ├── finalization_intent/<sha256>.json
│   ├── operations/
│   │   ├── model/<operation>/{intent,result}.json
│   │   └── experiment/<repeat>-<side>/{intent,result}.json
│   └── executions/<repeat>-<side>/
│       ├── manifest.json
│       └── files/<manifest-bound execution files>
└── executions/<isolated-run>/
    ├── pytest.ini
    ├── authorization.feature
    ├── test_authorization.py
    ├── observation.json
    ├── observation.json.events.jsonl
    ├── pytest-outcome.json
    ├── pytest.stdout.txt
    └── pytest.stderr.txt
```

The records retain the prepared and approved meaning, structured plan,
generated source, validation decision, model provenance, paired raw
observations, controls, timings, hashes, lifecycle events, and exclusive final
record. Recorder-owned execution copies are immutable and manifest-bound;
top-level execution directories are isolated subprocess working copies. Unknown
in-flight external outcomes are not guessed or silently repeated.

Inspect the exact absolute run path shown in the UI's **Local artifacts**
expander:

```bash
RUN_DIR='/absolute/path/copied-from-the-Local-artifacts-expander'
find "$RUN_DIR" -maxdepth 5 -type f | sort
python -m json.tool "$RUN_DIR/run_record.json"
```

Artifacts are ignored by Git. After retaining any records needed for an
experiment, remove only the reviewed run directory with an explicit path:

```bash
rm -rf -- '/absolute/path/to/repository/artifacts/run-reviewed-id'
```

Never place credentials, environment dumps, provider headers, or shell history
in the artifact directory.

## Roadmap

The near-term research sequence is:

1. Operate and evaluate Milestone 2 against a reviewed sample of real OpenMRS
   Core pull requests.
2. Execute approved tests against exact real base and PR revisions in isolated
   environments.
3. Surface the evidence and provisional expert review as a GitHub check.
4. Build the expert-labeled real-PR and seeded/mutation study corpus.
5. Run the preregistered, sample-size-justified publication evaluation.

Until those stages are implemented and evaluated, TriageGuard should be
described as a local research prototype that demonstrates one auditable,
evidence-bounded authorization experiment.
