# Synthetic OpenMRS-shaped PR fixture

This directory is a deterministic development and integration-test fixture.
It is **not** evidence about a real OpenMRS Core pull request and was never
fetched from GitHub. Pull request `900000001` is reserved synthetic input.

The Git bundle contains four primary commits:

- **M**: `PatientService.purgePatient` has a delete-privilege annotation.
- **B**: current `main` adds only an audit-documentation change.
- **H**: the synthetic PR removes the delete-privilege annotation from M.
- **C**: a merge commit with first parent B and second parent H.

The bundle also contains a no-drift merge preview used by the explicit
`M == B` acceptance case. Its first parent is M, its second parent is H, and
its tree matches the proposed head. The resulting M→B comparison is a
canonical `unchanged` artifact; it is never represented by a missing diff.

The model JSON files are prerecorded structured replay templates. Their
placeholder values are deterministically replaced only with the snapshot,
context, evidence-envelope, integration-anchor, and human-review identities
already present in the corresponding live-shaped model request. No LLM call
and no network request occurs. Replay supplies model outputs only; it does not
bypass envelope hashing, request sizing, citation checks, testability, or
Gherkin validation.

The fixture can demonstrate an approved Gherkin scenario only after locally
validated frozen-code testability. It can also demonstrate explicit non-risk
outcomes. In every case, the result is development evidence about TriageGuard;
it never labels the synthetic change safe or proves anything about OpenMRS.

When a real or replayed review cannot find enough relevant code among its four
already frozen photographs, Milestone 2 records **insufficient frozen code
evidence to design an executable scenario**. That is an honest boundary of the
available code evidence, not a safety conclusion.
