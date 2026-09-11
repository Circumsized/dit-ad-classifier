"""Verify prose cleanup against a saved functional patch.

Compares the current tree with HEAD after removing docstrings and applying an
allowlist for the functional hardening completed before the prose pass.
Comments never reach the AST, so any other difference is unintended.
"""

from __future__ import annotations

import ast
import subprocess
import sys


def strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return tree


def remove_allowed_hardening(tree: ast.AST, filename: str) -> ast.AST:
    """Normalize the two functional changes that preceded this prose pass."""

    if filename == "dit/evaluation/reporting.py":
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_json_safe":
                node.body = [
                    statement
                    for statement in node.body
                    if not (
                        isinstance(statement, ast.If)
                        and isinstance(statement.test, ast.Call)
                        and isinstance(statement.test.func, ast.Name)
                        and statement.test.func.id == "isinstance"
                        and len(statement.test.args) == 2
                        and isinstance(statement.test.args[1], ast.Attribute)
                        and statement.test.args[1].attr in {"bool_", "integer"}
                    )
                ]

    if filename == "dit/evaluation/runner.py":
        helper = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_ensure_complete_oof"
            ),
            None,
        )
        tree.body = [node for node in tree.body if node is not helper]
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_classical":
                for index, statement in enumerate(node.body):
                    if (
                        isinstance(statement, ast.Expr)
                        and isinstance(statement.value, ast.Call)
                        and isinstance(statement.value.func, ast.Name)
                        and statement.value.func.id == "_ensure_complete_oof"
                    ):
                        replacement = ast.parse(
                            "valid = (oof_pred >= 0) & np.all(np.isfinite(oof_prob), axis=1)\n"
                            "if not np.all(valid):\n"
                            "    missing = np.flatnonzero(~valid).tolist()\n"
                            "    raise RuntimeError(f'cross-validation did not predict every subject: {missing[:10]}')"
                        ).body
                        node.body[index : index + 1] = replacement
                        break
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_metric_ensemble":
                node.body = [
                    statement
                    for statement in node.body
                    if not (
                        isinstance(statement, ast.Expr)
                        and isinstance(statement.value, ast.Call)
                        and isinstance(statement.value.func, ast.Name)
                        and statement.value.func.id == "_ensure_complete_oof"
                    )
                ]
    return tree


def main() -> int:
    listing = subprocess.run(
        ["git", "diff", "--name-only", "--", "dit/"],
        capture_output=True,
        text=True,
        check=True,
    )
    changed = [name for name in listing.stdout.split() if name.endswith(".py")]

    differing: list[str] = []
    for name in changed:
        original = subprocess.run(
            ["git", "show", f"HEAD:{name}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        with open(name, encoding="utf-8") as handle:
            current = handle.read()

        before = ast.dump(strip_docstrings(ast.parse(original)))
        current_tree = strip_docstrings(ast.parse(current))
        after = ast.dump(remove_allowed_hardening(current_tree, name))
        if before == after:
            print(f"{'SAME':18} {name}")
        else:
            differing.append(name)
            print(f"{'*** DIFFERS ***':18} {name}")

    print()
    if differing:
        print("files with code or literal changes:", differing)
        return 1
    print(f"all {len(changed)} files: only docstrings/comments changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
