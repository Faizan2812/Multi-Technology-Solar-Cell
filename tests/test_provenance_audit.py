"""Regression test: the database must pass the integrity audit."""
from physics.provenance_audit import audit
def test_database_audit_clean():
    db, findings = audit()
    errs = [f for f in findings if f["severity"]=="ERROR"]
    assert not errs, f"audit errors: {errs}"
def test_no_tombstones():
    db,_ = audit()
    assert not any(k.startswith("_REMOVED_") for k in db["_references"])
