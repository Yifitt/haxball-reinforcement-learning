from __future__ import annotations

from scripts.check_documentation import audit_repository, checklist


def test_readme_documents_every_maintained_executable_script() -> None:
    audit = audit_repository()
    assert not audit.missing_from_readme, (
        "Document each new executable in README.md or add a justified vendored-only "
        f"exception to DOCUMENTATION_IGNORE: {audit.missing_from_readme}"
    )


def test_readme_has_no_stale_documented_script_paths() -> None:
    audit = audit_repository()
    assert not audit.stale_documented_paths, (
        "Remove or correct stale README script paths: "
        f"{audit.stale_documented_paths}"
    )


def test_documentation_helper_exposes_definition_of_done_checklist() -> None:
    text = checklist()
    assert "every new executable" in text.lower()
    assert "secrets" in text.lower()
