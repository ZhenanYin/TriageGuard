# Synthetic OpenMRS-shaped PR fixture

This directory is a deterministic development and integration-test fixture.
It is **not** evidence about a real OpenMRS Core pull request and was never
fetched from GitHub. Pull request `900000001` is reserved synthetic input.

The Git bundle contains four commits:

- **M**: `PatientService.purgePatient` has a delete-privilege annotation.
- **B**: current `main` adds only an audit-documentation change.
- **H**: the synthetic PR removes the delete-privilege annotation from M.
- **C**: a merge commit with first parent B and second parent H.

The model JSON files are prerecorded structured replay templates. Their
placeholder values are deterministically replaced only with the snapshot,
context, integration-anchor, and human-review identities already present in
the corresponding model request. No LLM call and no network request occurs.
