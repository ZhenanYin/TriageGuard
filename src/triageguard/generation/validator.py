"""Deterministic AST and fidelity validation for model-rendered pytest code."""

from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from triageguard.contracts.gherkin import validate_gherkin_alignment
from triageguard.domain.models import RiskContract, TestPlan
from triageguard.generation.code_generator import ALLOWED_IMPORTS
from triageguard.generation.planner import PlanValidationError, validate_test_plan
from triageguard.generation.primitives import PRIMITIVE_CATALOG

_GHERKIN_STEP = re.compile(r"^\s*(Given|When|Then|And|But|\*)\s+(.+?)\s*$")
_DECORATOR_NAMES = {"given", "when", "then"}
_RUNTIME_METHOD_TO_PRIMITIVE = {
    definition.runtime_helper.rsplit(".", 1)[1]: name
    for name, definition in PRIMITIVE_CATALOG.items()
}
_FORBIDDEN_NAMES = frozenset({"eval", "exec", "compile", "__import__"})
_FORBIDDEN_ROOTS = frozenset(
    {"subprocess", "requests", "httpx", "urllib", "socket", "shlex"}
)
_FORBIDDEN_OS_CALLS = frozenset(
    {
        "os.system",
        "os.popen",
        "os.fork",
        "os.spawnl",
        "os.spawnle",
        "os.spawnlp",
        "os.spawnlpe",
        "os.spawnv",
        "os.spawnve",
        "os.spawnvp",
        "os.spawnvpe",
    }
)
_SKIP_NAMES = frozenset({"skip", "skipif", "xfail", "importorskip"})
_APPROVED_ENVIRONMENT_KEYS = frozenset(
    {
        "OPENMRS_BASE_URL",
        "OPENMRS_USERNAME",
        "OPENMRS_PASSWORD",
        "TRIAGEGUARD_OBSERVATION_PATH",
    }
)
_PROTECTED_METHODS = frozenset(_RUNTIME_METHOD_TO_PRIMITIVE)
_PROTECTED_SYMBOLS = frozenset(
    {
        "ObservationWriter",
        "OpenMrsTestClient",
        "given",
        "os",
        "pathlib",
        "pytest",
        "pytest_bdd",
        "scenario",
        "scenarios",
        "then",
        "when",
    }
) | _PROTECTED_METHODS
_PROTECTED_RUNTIME_ROOTS = _PROTECTED_SYMBOLS | frozenset(
    {"observation_writer", "openmrs_client"}
)
_PROTECTED_COLLECTION_NAMES = frozenset({"__test__", "pytestmark"})
_ALLOWED_DIRECT_CALLS = frozenset(
    {
        "OpenMrsTestClient",
        "ObservationWriter",
        "Path",
        "given",
        "pytest.fixture",
        "scenario",
        "test_context.get",
        "then",
        "when",
    }
)


class CodeValidationReport(BaseModel):
    """Independent static decision over generated source, never model claims."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approved: bool
    reason_codes: list[str]
    implemented_steps: list[str]
    used_primitives: list[str]
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def validate_generated_code(
    code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
) -> CodeValidationReport:
    """Approve only code whose AST preserves the already-approved experiment."""
    reasons: list[str] = []
    alignment = validate_gherkin_alignment(contract, gherkin)
    if not alignment.approved:
        for reason in alignment.reason_codes:
            _add_reason(reasons, f"gherkin_{reason}")
    try:
        validate_test_plan(contract, plan)
    except PlanValidationError as error:
        for reason in error.reason_codes:
            _add_reason(reasons, f"plan_{reason}")

    try:
        tree = ast.parse(code)
    except SyntaxError:
        _add_reason(reasons, "syntax_invalid")
        return _report(code, reasons, [], [])

    _validate_imports(tree, reasons)
    _validate_forbidden_constructs(tree, reasons)
    _validate_scenario_binding(tree, contract, reasons)
    _validate_collection_bindings(tree, reasons)
    _validate_pytest_extensions(tree, reasons)
    _validate_protected_symbols(tree, reasons)
    _validate_protected_symbol_uses(tree, reasons)
    _validate_yield_usage(tree, reasons)

    expected_steps = _expected_steps(gherkin)
    implemented, decorated_functions = _implemented_steps(tree)
    if Counter(implemented) != Counter(text for _, text in expected_steps):
        expected_texts = Counter(text for _, text in expected_steps)
        implemented_texts = Counter(implemented)
        if any(implemented_texts[text] < count for text, count in expected_texts.items()):
            _add_reason(reasons, "step_implementation_missing")
        if any(implemented_texts[text] > expected_texts[text] for text in implemented_texts):
            _add_reason(reasons, "step_implementation_extra")
    if _decorator_signatures(decorated_functions) != expected_steps:
        _add_reason(reasons, "step_decorator_mismatch")

    runtime_receivers = _runtime_receiver_names(tree)
    _validate_call_allowlist(tree, reasons)
    used_primitives = _validate_runtime_calls(tree, runtime_receivers, reasons)
    _validate_required_primitives(plan, used_primitives, reasons)
    _validate_runtime_configuration(tree, reasons)
    _validate_plan_dataflow(tree, plan, reasons)
    _validate_plan_phase_fidelity(tree, contract, plan, reasons)
    _validate_observation_writes(tree, plan, contract, reasons)
    _validate_oracles(contract, plan, decorated_functions, reasons)
    _validate_control_and_cleanup(tree, contract, reasons)

    return _report(code, reasons, implemented, used_primitives)


def _validate_imports(tree: ast.AST, reasons: list[str]) -> None:
    imported_runtime_names: set[str] = set()
    imported_pytest_bdd = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    _add_reason(reasons, "forbidden_import")
                if alias.asname is not None:
                    _add_reason(reasons, "import_alias_forbidden")
                if alias.name == "pytest_bdd":
                    imported_pytest_bdd = True
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_IMPORTS:
                _add_reason(reasons, "forbidden_import")
            if node.module == "os" and any(
                alias.name not in {"environ", "getenv"} for alias in node.names
            ):
                _add_reason(reasons, "forbidden_import")
            if node.module == "triageguard.runtime" and any(
                alias.name not in {"OpenMrsTestClient", "ObservationWriter"}
                for alias in node.names
            ):
                _add_reason(reasons, "forbidden_import")
            if any(alias.name == "*" for alias in node.names):
                _add_reason(reasons, "wildcard_import_forbidden")
            if any(alias.asname is not None for alias in node.names):
                _add_reason(reasons, "import_alias_forbidden")
            if node.module == "pytest_bdd":
                imported_pytest_bdd = True
            if node.module == "triageguard.runtime":
                imported_runtime_names.update(alias.name for alias in node.names)
    if not imported_pytest_bdd:
        _add_reason(reasons, "pytest_bdd_import_missing")
    if not {"OpenMrsTestClient", "ObservationWriter"}.issubset(imported_runtime_names):
        _add_reason(reasons, "runtime_import_missing")


def _validate_forbidden_constructs(tree: ast.AST, reasons: list[str]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Try, ast.TryStar)) and node.handlers:
            _add_reason(reasons, "exception_swallowing")
        if isinstance(node, ast.Attribute) and node.attr in _SKIP_NAMES:
            _add_reason(reasons, "skip_forbidden")
        if isinstance(node, ast.Name) and node.id in _SKIP_NAMES:
            _add_reason(reasons, "skip_forbidden")
        if isinstance(node, ast.Attribute) and _is_dunder_name(node.attr):
            _add_reason(reasons, "dunder_attribute_forbidden")
        if isinstance(node, ast.Call):
            path = _call_path(node.func)
            names = set(path.split("."))
            root = path.split(".", 1)[0]
            if (
                root in _FORBIDDEN_ROOTS
                or path in _FORBIDDEN_NAMES
                or path in _FORBIDDEN_OS_CALLS
            ):
                _add_reason(reasons, "forbidden_call")
            if names & _SKIP_NAMES:
                _add_reason(reasons, "skip_forbidden")
            if path == "os.getenv" and (len(node.args) > 1 or node.keywords):
                _add_reason(reasons, "environment_default_forbidden")
            if path in {"os.environ.get", "os.environ.setdefault"} and (
                len(node.args) > 1 or node.keywords or path.endswith("setdefault")
            ):
                _add_reason(reasons, "environment_default_forbidden")


def _validate_call_allowlist(tree: ast.AST, reasons: list[str]) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path = _call_path(node.func)
        if path in _ALLOWED_DIRECT_CALLS:
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in (
            set(_RUNTIME_METHOD_TO_PRIMITIVE) | {"get"}
        ):
            continue
        _add_reason(reasons, "forbidden_call")


def _validate_scenario_binding(
    tree: ast.Module, contract: RiskContract, reasons: list[str]
) -> None:
    scenario_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _call_path(node.func) in {"scenario", "scenarios"}
    ]
    intended = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_patient_delete_authorization"
    ]
    collected_tests = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        and not _has_decorator(node, "pytest.fixture")
    ]
    if len(scenario_calls) != 1 or len(intended) != 1 or collected_tests != intended:
        _add_reason(reasons, "scenario_binding_invalid")
        return

    function = intended[0]
    if (
        function.args.posonlyargs
        or function.args.args
        or function.args.kwonlyargs
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or len(function.body) != 1
        or not isinstance(function.body[0], ast.Pass)
        or len(function.decorator_list) != 1
        or function.decorator_list[0] is not scenario_calls[0]
        or not _is_exact_scenario_decorator(scenario_calls[0], contract.contract_id)
    ):
        _add_reason(reasons, "scenario_binding_invalid")


def _is_exact_scenario_decorator(call: ast.Call, scenario_title: str) -> bool:
    return (
        _call_path(call.func) == "scenario"
        and len(call.args) == 2
        and not call.keywords
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "authorization.feature"
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == scenario_title
    )


def _validate_collection_bindings(tree: ast.AST, reasons: list[str]) -> None:
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(
            node,
            (
                ast.AnnAssign,
                ast.AugAssign,
                ast.NamedExpr,
                ast.For,
                ast.AsyncFor,
                ast.comprehension,
            ),
        ):
            targets = [node.target]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            targets = [
                item.optional_vars
                for item in node.items
                if item.optional_vars is not None
            ]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        if any(
            _bound_names(target) & _PROTECTED_COLLECTION_NAMES
            or _target_attribute_names(target) & _PROTECTED_COLLECTION_NAMES
            for target in targets
        ):
            _add_reason(reasons, "test_collection_disabled")
            return


def _validate_pytest_extensions(tree: ast.AST, reasons: list[str]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name.startswith("pytest_")
        ):
            _add_reason(reasons, "pytest_extension_forbidden")
            return
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = [node.target]
        if any("pytest_plugins" in _bound_names(target) for target in targets):
            _add_reason(reasons, "pytest_extension_forbidden")
            return


def _target_attribute_names(target: ast.expr) -> set[str]:
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(
            *(_target_attribute_names(element) for element in target.elts)
        )
    if isinstance(target, ast.Starred):
        return _target_attribute_names(target.value)
    if isinstance(target, ast.Attribute):
        return {target.attr}
    return set()


def _validate_yield_usage(tree: ast.Module, reasons: list[str]) -> None:
    control_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "authorized_control_and_cleanup"
    ]
    allowed_yield: ast.Yield | None = None
    if len(control_functions) == 1 and control_functions[0].body:
        first = control_functions[0].body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Yield)
            and first.value.value is None
        ):
            allowed_yield = first.value

    yields = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Yield, ast.YieldFrom))
    ]
    if any(node is not allowed_yield for node in yields):
        _add_reason(reasons, "generated_yield_forbidden")


def _is_dunder_name(name: str) -> bool:
    return len(name) >= 4 and name.startswith("__") and name.endswith("__")


def _has_decorator(
    function: ast.FunctionDef | ast.AsyncFunctionDef, decorator_path: str
) -> bool:
    return any(
        _call_path(decorator.func) == decorator_path
        if isinstance(decorator, ast.Call)
        else _call_path(decorator) == decorator_path
        for decorator in function.decorator_list
    )


def _validate_protected_symbols(tree: ast.AST, reasons: list[str]) -> None:
    rebound = _has_forbidden_rebinding(tree, set(_PROTECTED_SYMBOLS))
    rebound = rebound or _imports_rebind_protected_symbols(tree)
    rebound = rebound or not _has_exact_runtime_fixtures(tree)
    runtime_bindings = {"observation_writer", "openmrs_client", "test_context"}
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(
            node,
            (
                ast.AnnAssign,
                ast.AugAssign,
                ast.NamedExpr,
                ast.For,
                ast.AsyncFor,
                ast.comprehension,
            ),
        ):
            targets = [node.target]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            targets = [
                item.optional_vars
                for item in node.items
                if item.optional_vars is not None
            ]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        if any(_bound_names(target) & _PROTECTED_SYMBOLS for target in targets):
            rebound = True
        if any(_bound_names(target) & runtime_bindings for target in targets):
            rebound = True
        if any(_is_protected_attribute_target(target) for target in targets):
            rebound = True
        if any(_is_protected_runtime_mutation(target) for target in targets):
            rebound = True
        if isinstance(node, ast.Call) and _call_path(node.func) in {
            "setattr",
            "delattr",
        }:
            rebound = True
    if rebound:
        _add_reason(reasons, "protected_symbol_rebound")


def _validate_protected_symbol_uses(tree: ast.Module, reasons: list[str]) -> None:
    approved_loads = _approved_protected_loads(tree)
    if any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in _PROTECTED_SYMBOLS
        and id(node) not in approved_loads
        for node in ast.walk(tree)
    ):
        _add_reason(reasons, "protected_symbol_propagated")


def _approved_protected_loads(tree: ast.Module) -> set[int]:
    approved: set[int] = set()
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    fixture_constructors = {
        "openmrs_client": "OpenMrsTestClient",
        "observation_writer": "ObservationWriter",
    }
    for function_name, constructor_name in fixture_constructors.items():
        function = functions.get(function_name)
        if function is None or len(function.body) != 1:
            continue
        statement = function.body[0]
        if (
            isinstance(statement, ast.Return)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == constructor_name
        ):
            approved.add(id(statement.value.func))

    _approve_canonical_environment_loads(functions, approved)

    fixture_names = {
        "authorized_control_and_cleanup",
        "observation_writer",
        "openmrs_client",
        "test_context",
    }
    for function in functions.values():
        for decorator in function.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id in _DECORATOR_NAMES | {"scenario"}
            ):
                approved.add(id(decorator.func))
            if function.name in fixture_names:
                fixture_attribute = (
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                if (
                    isinstance(fixture_attribute, ast.Attribute)
                    and fixture_attribute.attr == "fixture"
                    and isinstance(fixture_attribute.value, ast.Name)
                    and fixture_attribute.value.id == "pytest"
                ):
                    approved.add(id(fixture_attribute.value))
    return approved


def _approve_canonical_environment_loads(
    functions: dict[str, ast.FunctionDef], approved: set[int]
) -> None:
    client = _single_return_call(functions.get("openmrs_client"))
    if client is not None and _call_path(client.func) == "OpenMrsTestClient":
        expected_keys = {
            "base_url": "OPENMRS_BASE_URL",
            "username": "OPENMRS_USERNAME",
            "password": "OPENMRS_PASSWORD",
        }
        keywords = {keyword.arg: keyword.value for keyword in client.keywords}
        for name, key in expected_keys.items():
            _approve_os_environment_name(keywords.get(name), key, approved)

    writer = _single_return_call(functions.get("observation_writer"))
    if (
        writer is not None
        and _call_path(writer.func) == "ObservationWriter"
        and len(writer.args) == 1
    ):
        _approve_os_environment_name(
            writer.args[0], "TRIAGEGUARD_OBSERVATION_PATH", approved
        )


def _single_return_call(function: ast.FunctionDef | None) -> ast.Call | None:
    if function is None or len(function.body) != 1:
        return None
    statement = function.body[0]
    if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Call):
        return statement.value
    return None


def _approve_os_environment_name(
    expression: ast.expr | None, key: str, approved: set[int]
) -> None:
    if (
        isinstance(expression, ast.Subscript)
        and isinstance(expression.ctx, ast.Load)
        and _literal_subscript_key(expression) == key
        and isinstance(expression.value, ast.Attribute)
        and expression.value.attr == "environ"
        and isinstance(expression.value.value, ast.Name)
        and expression.value.value.id == "os"
    ):
        approved.add(id(expression.value.value))


def _imports_rebind_protected_symbols(tree: ast.AST) -> bool:
    canonical_sources = {
        "ObservationWriter": "triageguard.runtime",
        "OpenMrsTestClient": "triageguard.runtime",
        "given": "pytest_bdd",
        "scenario": "pytest_bdd",
        "then": "pytest_bdd",
        "when": "pytest_bdd",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                if alias.asname in _PROTECTED_SYMBOLS:
                    return True
                if bound in {"os", "pathlib", "pytest", "pytest_bdd"} and (
                    alias.name != bound
                ):
                    return True
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound = alias.asname or alias.name
                if alias.asname in _PROTECTED_SYMBOLS:
                    return True
                if bound in canonical_sources and node.module != canonical_sources[bound]:
                    return True
    return False


def _has_exact_runtime_fixtures(tree: ast.AST) -> bool:
    functions = {
        name: [
            node
            for node in getattr(tree, "body", [])
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        for name in ("observation_writer", "openmrs_client", "test_context")
    }
    if any(len(matches) != 1 for matches in functions.values()):
        return False
    context = functions["test_context"][0]
    client = functions["openmrs_client"][0]
    writer = functions["observation_writer"][0]
    return (
        _is_plain_fixture(context)
        and len(context.body) == 1
        and isinstance(context.body[0], ast.Return)
        and isinstance(context.body[0].value, ast.Dict)
        and not context.body[0].value.keys
        and _is_plain_fixture(client)
        and _returns_direct_constructor(client, "OpenMrsTestClient")
        and _is_plain_fixture(writer)
        and _returns_direct_constructor(writer, "ObservationWriter")
    )


def _is_plain_fixture(function: ast.FunctionDef) -> bool:
    return (
        len(function.decorator_list) == 1
        and _call_path(function.decorator_list[0]) == "pytest.fixture"
        and not function.args.posonlyargs
        and not function.args.args
        and not function.args.kwonlyargs
        and function.args.vararg is None
        and function.args.kwarg is None
    )


def _returns_direct_constructor(function: ast.FunctionDef, name: str) -> bool:
    return (
        len(function.body) == 1
        and isinstance(function.body[0], ast.Return)
        and isinstance(function.body[0].value, ast.Call)
        and _call_path(function.body[0].value.func) == name
    )


def _is_protected_attribute_target(target: ast.expr) -> bool:
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_is_protected_attribute_target(element) for element in target.elts)
    if isinstance(target, ast.Starred):
        return _is_protected_attribute_target(target.value)
    if not isinstance(target, ast.Attribute):
        return False
    return (
        _root_name(target) in _PROTECTED_SYMBOLS
        or target.attr in _PROTECTED_METHODS
    )


def _is_protected_runtime_mutation(target: ast.expr) -> bool:
    return isinstance(target, (ast.Attribute, ast.Subscript)) and (
        _mutation_root_name(target) in _PROTECTED_RUNTIME_ROOTS
    )


def _mutation_root_name(expression: ast.expr) -> str:
    while isinstance(expression, (ast.Attribute, ast.Subscript)):
        expression = expression.value
    return expression.id if isinstance(expression, ast.Name) else ""


def _expected_steps(gherkin: str) -> list[tuple[str, str]]:
    expected: list[tuple[str, str]] = []
    phase: str | None = None
    for line in gherkin.splitlines():
        match = _GHERKIN_STEP.match(line)
        if match is None:
            continue
        keyword, text = match.groups()
        if keyword == "Given":
            phase = "given"
        elif keyword == "When":
            phase = "when"
        elif keyword == "Then":
            phase = "then"
        if keyword == "*" or phase is None:
            expected.append(("invalid", text))
        else:
            expected.append((phase, text))
    return expected


def _implemented_steps(
    tree: ast.Module,
) -> tuple[list[str], list[tuple[str, str, ast.FunctionDef | ast.AsyncFunctionDef]]]:
    implemented: list[str] = []
    decorated: list[tuple[str, str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            name = _call_path(decorator.func)
            if name not in _DECORATOR_NAMES or len(decorator.args) != 1:
                continue
            value = decorator.args[0]
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                implemented.append(value.value)
                decorated.append((name, value.value, node))
    return implemented, decorated


def _decorator_signatures(
    decorated: list[tuple[str, str, ast.FunctionDef | ast.AsyncFunctionDef]],
) -> list[tuple[str, str]]:
    return [(name, text) for name, text, _ in decorated]


def _runtime_receiver_names(tree: ast.AST) -> set[str]:
    receivers = {"OpenMrsTestClient", "ObservationWriter"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            isinstance(child, ast.Return)
            and isinstance(child.value, ast.Call)
            and _call_path(child.value.func)
            in {"OpenMrsTestClient", "ObservationWriter"}
            for child in ast.walk(node)
        ):
            receivers.add(node.name)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, ast.Call) and _call_path(value.func) in {
                "OpenMrsTestClient",
                "ObservationWriter",
            }:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        receivers.add(target.id)
    return receivers


def _validate_runtime_calls(
    tree: ast.AST,
    runtime_receivers: set[str],
    reasons: list[str],
) -> list[str]:
    used: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        receiver = _root_name(node.func.value)
        method = node.func.attr
        if method in _RUNTIME_METHOD_TO_PRIMITIVE:
            if receiver not in runtime_receivers:
                _add_reason(reasons, "runtime_receiver_unknown")
                continue
            primitive = _RUNTIME_METHOD_TO_PRIMITIVE[method]
            used.append(primitive)
        elif receiver in runtime_receivers:
            _add_reason(reasons, "runtime_method_unknown")
    return used


def _validate_required_primitives(
    plan: TestPlan, used_primitives: list[str], reasons: list[str]
) -> None:
    required = [
        operation.primitive
        for operation in plan.givens
        + [plan.action]
        + plan.post_action
    ]
    required.extend(assertion.primitive for assertion in plan.assertions)
    for control in plan.controls:
        required.extend(operation.primitive for operation in control.operations)
        required.extend(assertion.primitive for assertion in control.assertions)
    required.extend(operation.primitive for operation in plan.cleanup)
    if Counter(required) != Counter(used_primitives):
        _add_reason(reasons, "planned_primitive_occurrence_mismatch")


def _validate_runtime_configuration(tree: ast.AST, reasons: list[str]) -> None:
    invalid = False
    mutated = False
    parents = _parent_map(tree)
    environment_reads: Counter[str] = Counter()

    if any(
        isinstance(node, ast.ImportFrom) and node.module == "os"
        for node in ast.walk(tree)
    ):
        invalid = True

    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
            if _call_path(node.value) == "os.environ":
                invalid = True
            if any(
                _is_sensitive_name(name) and _contains_string_literal(node.value)
                for target in targets
                for name in _bound_names(target)
            ):
                invalid = True
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        if any(_is_environment_target(target) for target in targets):
            mutated = True
            invalid = True

        if isinstance(node, ast.Subscript) and _call_path(node.value) == "os.environ":
            key = _literal_subscript_key(node)
            if not isinstance(node.ctx, ast.Load):
                mutated = True
                invalid = True
            elif key not in _APPROVED_ENVIRONMENT_KEYS:
                invalid = True
            else:
                environment_reads[key] += 1

        if isinstance(node, ast.Attribute) and _call_path(node) == "os.environ":
            parent = parents.get(node)
            if not (isinstance(parent, ast.Subscript) and parent.value is node):
                invalid = True

        if isinstance(node, ast.Call):
            path = _call_path(node.func)
            if path == "os.getenv" or path in {
                "os.putenv",
                "os.unsetenv",
                "os.environ.clear",
                "os.environ.pop",
                "os.environ.setdefault",
                "os.environ.update",
            }:
                invalid = True
                if path != "os.getenv":
                    mutated = True

    if environment_reads != Counter(
        {key: 1 for key in _APPROVED_ENVIRONMENT_KEYS}
    ):
        invalid = True

    constructors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_path(node.func) == "OpenMrsTestClient"
    ]
    if len(constructors) != 1:
        invalid = True
    else:
        keywords = {keyword.arg: keyword.value for keyword in constructors[0].keywords}
        expected_keys = {
            "base_url": "OPENMRS_BASE_URL",
            "username": "OPENMRS_USERNAME",
            "password": "OPENMRS_PASSWORD",
        }
        if set(keywords) != set(expected_keys) or any(
            _environment_read_key(keywords[name]) != expected_key
            for name, expected_key in expected_keys.items()
        ):
            invalid = True

    writers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_path(node.func) == "ObservationWriter"
    ]
    if (
        len(writers) != 1
        or len(writers[0].args) != 1
        or writers[0].keywords
        or _environment_read_key(writers[0].args[0])
        != "TRIAGEGUARD_OBSERVATION_PATH"
    ):
        invalid = True

    if mutated:
        _add_reason(reasons, "runtime_configuration_mutated")
    if invalid:
        _add_reason(reasons, "runtime_configuration_invalid")


def _literal_subscript_key(node: ast.Subscript) -> str | None:
    return (
        node.slice.value
        if isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
        else None
    )


def _environment_read_key(expression: ast.expr) -> str | None:
    if (
        isinstance(expression, ast.Subscript)
        and _call_path(expression.value) == "os.environ"
        and isinstance(expression.ctx, ast.Load)
    ):
        return _literal_subscript_key(expression)
    return None


def _is_sensitive_name(name: str) -> bool:
    lowered = name.casefold()
    return any(
        token in lowered
        for token in ("base_url", "credential", "password", "target_url", "username")
    )


def _contains_string_literal(expression: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Constant) and isinstance(node.value, str)
        for node in ast.walk(expression)
    )


def _validate_plan_dataflow(
    tree: ast.AST,
    plan: TestPlan,
    reasons: list[str],
) -> None:
    expected = _expected_client_calls(plan)
    call_targets = _direct_call_assignment_targets(tree)
    observed: Counter[tuple[str, tuple[object, ...], str | None]] = Counter()
    expected_methods = {method for method, _, _ in expected}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        if method not in expected_methods:
            continue
        signature = tuple(_argument_value(argument) for argument in node.args)
        observed[(method, signature, call_targets.get(id(node)))] += 1
        if node.keywords:
            _add_reason(reasons, "runtime_dataflow_mismatch")
    if observed != expected:
        _add_reason(reasons, "runtime_dataflow_mismatch")
    _validate_runtime_result_provenance(tree, expected, reasons)


def _expected_client_calls(
    plan: TestPlan,
) -> Counter[tuple[str, tuple[object, ...], str | None]]:
    expected: Counter[tuple[str, tuple[object, ...], str | None]] = Counter()
    operations = list(plan.givens) + [plan.action] + list(plan.post_action)
    for control in plan.controls:
        operations.extend(control.operations)
    operations.extend(plan.cleanup)
    for operation in operations:
        definition = PRIMITIVE_CATALOG.get(operation.primitive)
        if definition is None or not definition.runtime_helper.startswith(
            "OpenMrsTestClient."
        ):
            continue
        method = definition.runtime_helper.rsplit(".", 1)[1]
        signature = tuple(
            _plan_value(operation.inputs[name])
            for name in definition.input_types
        )
        target = (
            operation.captures[0].removeprefix("$")
            if len(operation.captures) == 1
            else None
        )
        expected[(method, signature, target)] += 1
    return expected


def _plan_value(value: str) -> str:
    return value.removeprefix("$")


def _validate_plan_phase_fidelity(
    tree: ast.Module,
    contract: RiskContract,
    plan: TestPlan,
    reasons: list[str],
) -> None:
    expected: Counter[tuple[str, str]] = Counter()
    expected.update(("setup", operation.primitive) for operation in plan.givens)
    expected[("action", plan.action.primitive)] += 1
    expected.update(
        ("post_action", operation.primitive) for operation in plan.post_action
    )
    expected.update(
        ("primary_assertion", assertion.primitive)
        for assertion in plan.assertions
    )
    for control in plan.controls:
        expected.update(
            ("control", operation.primitive) for operation in control.operations
        )
        expected.update(
            ("control_assertion", assertion.primitive)
            for assertion in control.assertions
        )
    expected.update(("cleanup", operation.primitive) for operation in plan.cleanup)

    parents = _parent_map(tree)
    observed: Counter[tuple[str, str]] = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        primitive = _RUNTIME_METHOD_TO_PRIMITIVE.get(node.func.attr)
        if primitive is None:
            continue
        function = _enclosing_function(node, parents)
        phase = _runtime_call_phase(
            node,
            function,
            parents,
            contract,
            plan,
            primitive,
        )
        observed[(phase, primitive)] += 1
    if (
        observed != expected
        or not _has_exact_setup_phase_order(tree, plan, parents)
        or not _has_exact_primary_phase_order(tree, contract, plan, parents)
    ):
        _add_reason(reasons, "planned_primitive_phase_mismatch")


def _has_exact_setup_phase_order(
    tree: ast.Module,
    plan: TestPlan,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    functions = sorted(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and _has_decorator(node, "given")
        ),
        key=lambda function: function.lineno,
    )
    calls = [
        node
        for function in functions
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _RUNTIME_METHOD_TO_PRIMITIVE
    ]
    calls.sort(key=lambda node: (node.lineno, node.col_offset))
    return [
        _RUNTIME_METHOD_TO_PRIMITIVE[call.func.attr] for call in calls
    ] == [operation.primitive for operation in plan.givens] and all(
        (function := _enclosing_function(call, parents)) is not None
        and isinstance(function, ast.FunctionDef)
        and _is_direct_function_call(call, function, parents)
        for call in calls
    )


def _has_exact_primary_phase_order(
    tree: ast.Module,
    contract: RiskContract,
    plan: TestPlan,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and _has_exact_step_decorator(node, "when", contract.action)
    ]
    if len(functions) != 1:
        return False
    function = functions[0]
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _RUNTIME_METHOD_TO_PRIMITIVE
    ]
    calls.sort(key=lambda node: (node.lineno, node.col_offset))
    expected = [plan.action.primitive]
    expected.extend(operation.primitive for operation in plan.post_action)
    expected.extend(assertion.primitive for assertion in plan.assertions)
    return [
        _RUNTIME_METHOD_TO_PRIMITIVE[call.func.attr] for call in calls
    ] == expected and all(
        _is_direct_function_call(call, function, parents) for call in calls
    )


def _is_direct_function_call(
    call: ast.Call,
    function: ast.FunctionDef,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    statement: ast.AST = call
    while statement in parents and parents[statement] is not function:
        statement = parents[statement]
    if parents.get(statement) is not function:
        return False
    return (
        isinstance(statement, ast.Assign) and statement.value is call
    ) or (
        isinstance(statement, ast.Expr) and statement.value is call
    )


def _runtime_call_phase(
    node: ast.Call,
    function: ast.FunctionDef | ast.AsyncFunctionDef | None,
    parents: dict[ast.AST, ast.AST],
    contract: RiskContract,
    plan: TestPlan,
    primitive: str,
) -> str:
    if function is None:
        return "unbound"
    if function.name == "authorized_control_and_cleanup":
        if _is_in_finally(node, parents):
            return "cleanup"
        if primitive in {assertion.primitive for control in plan.controls for assertion in control.assertions}:
            return "control_assertion"
        return "control"
    if _has_decorator(function, "given"):
        return "setup"
    if _has_exact_step_decorator(function, "when", contract.action):
        if primitive == plan.action.primitive:
            return "action"
        if primitive in {operation.primitive for operation in plan.post_action}:
            return "post_action"
        if primitive in {assertion.primitive for assertion in plan.assertions}:
            return "primary_assertion"
    return "unbound"


def _enclosing_function(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
    return None


def _validate_runtime_result_provenance(
    tree: ast.AST,
    expected_calls: Counter[tuple[str, tuple[object, ...], str | None]],
    reasons: list[str],
) -> None:
    call_targets = _direct_call_assignment_targets(tree)
    canonical_names = {
        target for _, _, target in expected_calls if target is not None
    }
    observed_calls: Counter[tuple[str, tuple[object, ...], str | None]] = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        if method not in {
            "create_patient",
            "login_as_actor",
            "delete_patient",
            "read_patient",
            "authorized_cleanup_patient",
        }:
            continue
        signature = tuple(_argument_value(argument) for argument in node.args)
        observed_calls[(method, signature, call_targets.get(id(node)))] += 1
    if observed_calls != expected_calls:
        _add_reason(reasons, "runtime_result_provenance_invalid")

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bound_names = _bound_names(target)
                if isinstance(target, ast.Name) and target.id in canonical_names:
                    if not _is_allowed_canonical_assignment(target.id, node.value):
                        _add_reason(reasons, "runtime_result_provenance_invalid")
                elif bound_names & canonical_names:
                    _add_reason(reasons, "runtime_result_provenance_invalid")
                if "test_context" in bound_names:
                    _add_reason(reasons, "runtime_result_provenance_invalid")
                key = _context_key(target)
                if key in canonical_names and not (
                    isinstance(node.value, ast.Name) and node.value.id == key
                ):
                    _add_reason(reasons, "runtime_result_provenance_invalid")
            if isinstance(node.value, ast.Name) and node.value.id == "test_context":
                _add_reason(reasons, "runtime_result_provenance_invalid")
    if _has_forbidden_rebinding(tree, canonical_names | {"test_context"}):
        _add_reason(reasons, "runtime_result_provenance_invalid")


def _direct_call_assignment_targets(tree: ast.AST) -> dict[int, str]:
    targets: dict[int, str] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
        ):
            targets[id(node.value)] = node.targets[0].id
    return targets


def _has_forbidden_rebinding(tree: ast.AST, protected: set[str]) -> bool:
    definition_protected = protected - {"test_context"}
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(
            node,
            (
                ast.AnnAssign,
                ast.AugAssign,
                ast.NamedExpr,
                ast.For,
                ast.AsyncFor,
                ast.comprehension,
            ),
        ):
            targets = [node.target]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            targets = [
                item.optional_vars
                for item in node.items
                if item.optional_vars is not None
            ]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        if any(
            _bound_names(target) & protected
            or _context_key(target) in protected
            for target in targets
        ):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            )
            if node.args.vararg is not None:
                arguments.append(node.args.vararg)
            if node.args.kwarg is not None:
                arguments.append(node.args.kwarg)
            if any(argument.arg in definition_protected for argument in arguments):
                return True
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                node.name in definition_protected
            ):
                return True
        if isinstance(node, ast.ClassDef) and node.name in definition_protected:
            return True
        if isinstance(node, ast.ExceptHandler) and node.name in protected:
            return True
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name in protected:
            return True
        if isinstance(node, ast.MatchMapping) and node.rest in protected:
            return True
    return False


def _bound_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_bound_names(element) for element in target.elts))
    if isinstance(target, ast.Starred):
        return _bound_names(target.value)
    return set()


def _is_allowed_canonical_assignment(name: str, value: ast.expr) -> bool:
    if (
        name == "control_patient_id"
        and isinstance(value, ast.Constant)
        and value.value is None
    ):
        return True
    if isinstance(value, ast.Call):
        if isinstance(value.func, ast.Attribute) and value.func.attr in {
            "create_patient",
            "login_as_actor",
            "delete_patient",
            "read_patient",
            "authorized_cleanup_patient",
        }:
            return True
        return (
            name == "patient_id"
            and _call_path(value.func) == "test_context.get"
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Constant)
            and value.args[0].value == "patient_id"
            and not value.keywords
        )
    return _context_key(value) == name


def _context_key(expression: ast.expr) -> str | None:
    if not isinstance(expression, ast.Subscript):
        return None
    if not isinstance(expression.value, ast.Name) or expression.value.id != "test_context":
        return None
    if isinstance(expression.slice, ast.Constant) and isinstance(
        expression.slice.value, str
    ):
        return expression.slice.value
    return None


def _is_environment_target(expression: ast.expr) -> bool:
    return isinstance(expression, ast.Subscript) and _call_path(
        expression.value
    ) == "os.environ"


def _validate_observation_writes(
    tree: ast.Module,
    plan: TestPlan,
    contract: RiskContract,
    reasons: list[str],
) -> None:
    expected_fields = {
        assertion.primitive: assertion.observed_field.removeprefix("$")
        for assertion in plan.assertions
    }
    for control in plan.controls:
        expected_fields.update(
            {
                assertion.primitive: assertion.observed_field.removeprefix("$")
                for assertion in control.assertions
            }
        )
    observed: dict[str, list[object]] = {
        primitive: [] for primitive in expected_fields
    }
    observation_calls: dict[str, list[ast.Call]] = {
        primitive: [] for primitive in expected_fields
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in observed
        ):
            observation_calls[node.func.attr].append(node)
            observed[node.func.attr].append(
                _argument_value(node.args[0]) if len(node.args) == 1 else None
            )
    if not observed.get("record_http_status"):
        _add_reason(reasons, "http_status_observation_missing")
    elif expected_fields.get("record_http_status") not in observed.get(
        "record_http_status", []
    ):
        _add_reason(reasons, "http_status_observation_mismatched")
    if not observed.get("record_patient_exists"):
        _add_reason(reasons, "patient_existence_observation_missing")
    elif (
        expected_fields.get("record_patient_exists")
        not in observed.get("record_patient_exists", [])
    ):
        _add_reason(reasons, "patient_existence_observation_mismatched")
    for primitive in (
        "record_control_patient_exists_before",
        "record_control_http_status",
        "record_control_patient_exists_after",
    ):
        if observed.get(primitive) != [expected_fields.get(primitive)]:
            _add_reason(reasons, "control_observation_mismatched")

    action_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and _has_exact_step_decorator(node, "when", contract.action)
    ]
    reachable = len(action_functions) == 1
    if reachable:
        action = action_functions[0]
        reachable = (
            len(action.body) >= 2
            and _is_direct_observation_statement(
                action.body[-2],
                "observation_writer.record_http_status",
                "delete_status",
            )
            and _is_direct_observation_statement(
                action.body[-1],
                "observation_writer.record_patient_exists",
                "patient_exists",
            )
            and len(observation_calls.get("record_http_status", [])) == 1
            and len(observation_calls.get("record_patient_exists", [])) == 1
            and not any(
                isinstance(node, (ast.Return, ast.Raise))
                for node in ast.walk(action)
            )
        )
        oracle_assertions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assert)
            and (
                _asserts_exact_value(node.test, "delete_status", 403)
                or _asserts_exact_value(node.test, "patient_exists", True)
            )
        ]
        if reachable:
            reachable = (
                len(oracle_assertions) == 2
                and action.body[-1].lineno
                < min(assertion.lineno for assertion in oracle_assertions)
            )
    if not reachable:
        _add_reason(reasons, "observation_unreachable")


def _has_exact_step_decorator(
    function: ast.FunctionDef, decorator_name: str, step_text: str
) -> bool:
    return any(
        isinstance(decorator, ast.Call)
        and _call_path(decorator.func) == decorator_name
        and len(decorator.args) == 1
        and not decorator.keywords
        and isinstance(decorator.args[0], ast.Constant)
        and decorator.args[0].value == step_text
        for decorator in function.decorator_list
    )


def _is_direct_observation_statement(
    statement: ast.stmt, call_path: str, argument_name: str
) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and _call_path(statement.value.func) == call_path
        and len(statement.value.args) == 1
        and not statement.value.keywords
        and isinstance(statement.value.args[0], ast.Name)
        and statement.value.args[0].id == argument_name
    )


def _validate_oracles(
    contract: RiskContract,
    plan: TestPlan,
    decorated_functions: list[
        tuple[str, str, ast.FunctionDef | ast.AsyncFunctionDef]
    ],
    reasons: list[str],
) -> None:
    denial = next(
        (
            assertion
            for assertion in plan.assertions
            if assertion.primitive == "record_http_status"
        ),
        None,
    )
    persistence = next(
        (
            assertion
            for assertion in plan.assertions
            if assertion.primitive == "record_patient_exists"
        ),
        None,
    )
    oracle_texts = [
        clause.strip()
        for clause in re.split(
            r"\s+and\s+", contract.secure_expectation, maxsplit=1
        )
    ]
    denial_text, persistence_text = (
        oracle_texts if len(oracle_texts) == 2 else ("", "")
    )

    denial_function = _decorated_function(decorated_functions, "then", denial_text)
    persistence_function = _decorated_function(
        decorated_functions, "then", persistence_text
    )
    if denial is None or denial_function is None or not _has_direct_assertion(
        denial_function, denial.observed_field, denial.expected_value
    ):
        _add_reason(reasons, "denial_assertion_missing")
    if persistence is None or persistence_function is None or not _has_direct_assertion(
        persistence_function,
        persistence.observed_field,
        persistence.expected_value,
    ):
        _add_reason(reasons, "patient_existence_assertion_missing")

    for function in (denial_function, persistence_function):
        if function is not None and any(
            isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom))
            for node in ast.walk(function)
        ):
            _add_reason(reasons, "assertion_return_bypass")


def _decorated_function(
    decorated: list[tuple[str, str, ast.FunctionDef | ast.AsyncFunctionDef]],
    decorator_name: str,
    text: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return next(
        (
            function
            for name, observed_text, function in decorated
            if name == decorator_name and observed_text == text
        ),
        None,
    )


def _has_direct_assertion(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    observed_field: str,
    expected_value: str | int | bool,
) -> bool:
    observed_name = observed_field.removeprefix("$")
    return any(
        isinstance(statement, ast.Assert)
        and _asserts_exact_value(statement.test, observed_name, expected_value)
        for statement in function.body
    )


def _asserts_exact_value(
    expression: ast.expr, observed_name: str, expected_value: str | int | bool
) -> bool:
    if not isinstance(expression, ast.Compare) or len(expression.ops) != 1:
        return False
    if not isinstance(expression.ops[0], (ast.Eq, ast.Is)):
        return False
    if len(expression.comparators) != 1:
        return False
    left = _value_key(expression.left)
    right = expression.comparators[0]
    if left == observed_name and isinstance(right, ast.Constant):
        return right.value == expected_value and type(right.value) is type(expected_value)
    right_key = _value_key(right)
    if right_key == observed_name and isinstance(expression.left, ast.Constant):
        return (
            expression.left.value == expected_value
            and type(expression.left.value) is type(expected_value)
        )
    return False


def _validate_control_and_cleanup(
    tree: ast.AST, contract: RiskContract, reasons: list[str]
) -> None:
    if not _has_exact_control_structure(tree):
        _add_reason(reasons, "authorized_control_structure_invalid")

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    login_actors = {
        node.args[0].value
        for node in calls
        if _call_path(node.func).endswith(".login_as_actor")
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    if contract.actor not in login_actors:
        _add_reason(reasons, "primary_actor_login_missing")
    control_functions = [
        function
        for function in tree.body
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(node, ast.Call)
            and _call_path(node.func).endswith(".login_as_actor")
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "administrator"
            for node in ast.walk(function)
        )
    ]
    if "administrator" not in login_actors or len(control_functions) != 1:
        _add_reason(reasons, "authorized_control_missing")
        return

    control_function = control_functions[0]
    status_assertions = _matching_assertions(
        control_function, "control_delete_status", 204
    )
    precondition_assertions = _matching_assertions(
        control_function, "control_patient_exists_before", True
    )
    persistence_assertions = _matching_assertions(
        control_function, "control_patient_exists", False
    )
    control_assertions = (
        precondition_assertions + status_assertions + persistence_assertions
    )
    if (
        len(precondition_assertions) != 1
        or len(status_assertions) != 1
        or len(persistence_assertions) != 1
    ):
        _add_reason(reasons, "authorized_control_assertion_missing")
    control_calls = [
        node
        for node in ast.walk(control_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        in {
            "create_patient",
            "login_as_actor",
            "delete_patient",
            "read_patient",
            "authorized_cleanup_patient",
        }
    ]
    cleanup_calls = [
        node
        for node in control_calls
        if node.func.attr == "authorized_cleanup_patient"
    ]
    if not cleanup_calls:
        _add_reason(reasons, "authorized_cleanup_missing")
        return

    parents = _parent_map(tree)
    yields = [
        node
        for node in ast.walk(control_function)
        if isinstance(node, (ast.Yield, ast.YieldFrom))
    ]
    if (
        not _is_autouse_fixture(control_function)
        or len(yields) != 1
        or any(node.lineno <= yields[0].lineno for node in control_calls)
        or any(
            _has_statically_false_ancestor(node, parents, control_function)
            for node in control_calls
        )
        or any(
            _has_statically_false_ancestor(node, parents, control_function)
            for node in control_assertions
        )
    ):
        _add_reason(reasons, "authorized_control_unreachable")
    if len(cleanup_calls) != 2 or any(
        not _is_in_finally(call, parents) for call in cleanup_calls
    ):
        _add_reason(reasons, "authorized_cleanup_not_guaranteed")


def _has_exact_control_structure(tree: ast.AST) -> bool:
    functions = [
        node
        for node in getattr(tree, "body", [])
        if isinstance(node, ast.FunctionDef)
        and node.name == "authorized_control_and_cleanup"
    ]
    if len(functions) != 1:
        return False
    function = functions[0]
    if (
        len(function.decorator_list) != 1
        or not _is_autouse_fixture(function)
        or [argument.arg for argument in function.args.args]
        != ["openmrs_client", "observation_writer", "test_context"]
        or function.args.posonlyargs
        or function.args.kwonlyargs
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or len(function.body) != 4
    ):
        return False

    yield_statement, patient_statement, control_initializer, control = function.body
    if not (
        isinstance(yield_statement, ast.Expr)
        and isinstance(yield_statement.value, ast.Yield)
        and yield_statement.value.value is None
        and _is_direct_assignment_call(
            patient_statement,
            "patient_id",
            "test_context.get",
            ("patient_id",),
        )
        and isinstance(control_initializer, ast.Assign)
        and len(control_initializer.targets) == 1
        and isinstance(control_initializer.targets[0], ast.Name)
        and control_initializer.targets[0].id == "control_patient_id"
        and isinstance(control_initializer.value, ast.Constant)
        and control_initializer.value.value is None
        and isinstance(control, ast.Try)
    ):
        return False

    if (
        control.handlers
        or control.orelse
        or len(control.body) != 11
        or len(control.finalbody) != 2
    ):
        return False
    (
        control_create,
        login,
        read_before,
        delete,
        read_after,
        record_before,
        record_status,
        record_after,
        precondition_assertion,
        status_assertion,
        persistence_assertion,
    ) = control.body
    control_cleanup, primary_cleanup_guard = control.finalbody
    return (
        _is_direct_assignment_call(
            control_create,
            "control_patient_id",
            "openmrs_client.create_patient",
            (),
        )
        and _is_direct_assignment_call(
            login,
            "control_actor_session",
            "openmrs_client.login_as_actor",
            ("administrator",),
        )
        and _is_direct_assignment_call(
            read_before,
            "control_patient_exists_before",
            "openmrs_client.read_patient",
            ("control_patient_id", "control_actor_session"),
        )
        and _is_direct_assignment_call(
            delete,
            "control_delete_status",
            "openmrs_client.delete_patient",
            ("control_patient_id", "control_actor_session"),
        )
        and _is_direct_assignment_call(
            read_after,
            "control_patient_exists",
            "openmrs_client.read_patient",
            ("control_patient_id", "control_actor_session"),
        )
        and _is_direct_observation_statement(
            record_before,
            "observation_writer.record_control_patient_exists_before",
            "control_patient_exists_before",
        )
        and _is_direct_observation_statement(
            record_status,
            "observation_writer.record_control_http_status",
            "control_delete_status",
        )
        and _is_direct_observation_statement(
            record_after,
            "observation_writer.record_control_patient_exists_after",
            "control_patient_exists",
        )
        and isinstance(precondition_assertion, ast.Assert)
        and _asserts_exact_value(
            precondition_assertion.test, "control_patient_exists_before", True
        )
        and isinstance(status_assertion, ast.Assert)
        and _asserts_exact_value(
            status_assertion.test, "control_delete_status", 204
        )
        and isinstance(persistence_assertion, ast.Assert)
        and _asserts_exact_value(
            persistence_assertion.test, "control_patient_exists", False
        )
        and isinstance(control_cleanup, ast.If)
        and _is_present_name_guard(control_cleanup.test, "control_patient_id")
        and not control_cleanup.orelse
        and len(control_cleanup.body) == 1
        and _is_direct_assignment_call(
            control_cleanup.body[0],
            "control_cleanup_complete",
            "openmrs_client.authorized_cleanup_patient",
            ("control_patient_id",),
        )
        and isinstance(primary_cleanup_guard, ast.If)
        and _is_patient_present_guard(primary_cleanup_guard.test)
        and not primary_cleanup_guard.orelse
        and len(primary_cleanup_guard.body) == 1
        and _is_direct_assignment_call(
            primary_cleanup_guard.body[0],
            "cleanup_complete",
            "openmrs_client.authorized_cleanup_patient",
            ("patient_id",),
        )
    )


def _is_direct_assignment_call(
    statement: ast.stmt,
    target_name: str,
    call_path: str,
    arguments: tuple[object, ...],
) -> bool:
    return (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == target_name
        and isinstance(statement.value, ast.Call)
        and _call_path(statement.value.func) == call_path
        and not statement.value.keywords
        and tuple(_argument_value(argument) for argument in statement.value.args)
        == arguments
    )


def _is_autouse_fixture(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call) or _call_path(decorator.func) != "pytest.fixture":
            continue
        return any(
            keyword.arg == "autouse"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in decorator.keywords
        )
    return False


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _has_statically_false_ancestor(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    boundary: ast.AST,
) -> bool:
    current = node
    while current in parents and current is not boundary:
        current = parents[current]
        if isinstance(
            current,
            (
                ast.AsyncFor,
                ast.For,
                ast.IfExp,
                ast.Lambda,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
                ast.While,
            ),
        ):
            return True
        if isinstance(current, ast.If) and not (
            _is_patient_present_guard(current.test)
            or _is_present_name_guard(current.test, "control_patient_id")
        ):
            return True
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            current is not boundary
        ):
            return True
    return False


def _is_patient_present_guard(expression: ast.expr) -> bool:
    return _is_present_name_guard(expression, "patient_id")


def _is_present_name_guard(expression: ast.expr, name: str) -> bool:
    return (
        isinstance(expression, ast.Compare)
        and isinstance(expression.left, ast.Name)
        and expression.left.id == name
        and len(expression.ops) == 1
        and isinstance(expression.ops[0], ast.IsNot)
        and len(expression.comparators) == 1
        and isinstance(expression.comparators[0], ast.Constant)
        and expression.comparators[0].value is None
    )


def _is_in_finally(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = node
    while current in parents:
        parent = parents[current]
        if isinstance(parent, (ast.Try, ast.TryStar)) and current in parent.finalbody:
            return True
        current = parent
    return False


def _value_key(expression: ast.expr) -> str | None:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Subscript):
        slice_value = expression.slice
        if isinstance(slice_value, ast.Constant) and isinstance(slice_value.value, str):
            return slice_value.value
    return None


def _argument_value(expression: ast.expr) -> object:
    key = _value_key(expression)
    if key is not None:
        return key
    if isinstance(expression, ast.Constant):
        return expression.value
    return None


def _is_required_environment_read(expression: ast.expr) -> bool:
    if isinstance(expression, ast.Subscript):
        return _call_path(expression.value) == "os.environ" and isinstance(
            expression.slice, ast.Constant
        )
    if isinstance(expression, ast.Call) and _call_path(expression.func) == "os.getenv":
        return len(expression.args) == 1 and not expression.keywords
    return False


def _matching_assertions(
    tree: ast.AST, observed_name: str, expected_value: str | int | bool
) -> list[ast.Assert]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
        and _asserts_exact_value(node.test, observed_name, expected_value)
    ]


def _call_path(expression: ast.expr) -> str:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        prefix = _call_path(expression.value)
        return f"{prefix}.{expression.attr}" if prefix else expression.attr
    return ""


def _root_name(expression: ast.expr) -> str:
    while isinstance(expression, ast.Attribute):
        expression = expression.value
    return expression.id if isinstance(expression, ast.Name) else ""


def _report(
    code: str,
    reasons: list[str],
    implemented_steps: list[str],
    used_primitives: list[str],
) -> CodeValidationReport:
    return CodeValidationReport(
        approved=not reasons,
        reason_codes=reasons,
        implemented_steps=implemented_steps,
        used_primitives=used_primitives,
        code_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
    )


def _add_reason(reason_codes: list[str], reason_code: str) -> None:
    if reason_code not in reason_codes:
        reason_codes.append(reason_code)
