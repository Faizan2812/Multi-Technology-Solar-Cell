"""
provenance_audit.py — integrity auditor for the material database.

Checks the database against the flag classes raised in the integrity assessment
and prevents their silent recurrence. Operates on the repository's native schema
(_references: {citation, doi, confidence, verified}; parameters: {value, source,
confidence, ...}). Returns structured findings; an empty ERROR list means the
database is clean.

Run:  python physics/provenance_audit.py
"""
import json, os, re
from collections import defaultdict

DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")
FORBIDDEN = [r"^_REMOVED_", r"^_DELETED_", r"\bin preparation\b", r"\bto be published\b",
             r"\bpersonal communication\b", r"\bphantom\b"]
VENUE_TOKENS = {"science": "science", "nature": "nature", "nat_materials": "nat. mater",
                "natcomm": "nat. commun", "jacs": "j. am. chem", "jem": "j. electron. mater",
                "acs_omega": "acs omega", "aem": "adv. energy mater", "ees": "energy environ"}
CONF = {"HIGH", "MEDIUM", "LOW"}


def audit(db_path=None):
    if db_path is None:
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "materials_database.json")
    db = json.load(open(db_path))
    refs = db["_references"]; refkeys = set(refs)
    findings = []
    def add(flag, sev, where, msg): findings.append(dict(flag=flag, severity=sev, where=where, message=msg))

    # F1 tombstones / forbidden patterns
    for k, r in refs.items():
        for pat in FORBIDDEN:
            if re.search(pat, k, re.I) or re.search(pat, str(r.get("citation", "")), re.I):
                add("F1-integrity", "ERROR", k, f"forbidden pattern '{pat}'")
    # F2 DOI format + duplicates with conflicting keys
    by_doi = defaultdict(list)
    for k, r in refs.items():
        doi = r.get("doi")
        if doi and doi not in ("UNKNOWN", "None", None):
            if not DOI_RE.match(doi):
                add("F2-doi", "ERROR", k, f"malformed DOI '{doi}'")
            by_doi[doi].append(k)
    for doi, ks in by_doi.items():
        if len(ks) > 1:
            add("F2-doi", "WARN", ", ".join(ks), f"DOI {doi} shared by {len(ks)} keys")
    # F3 venue-key mismatch
    for k, r in refs.items():
        cit = str(r.get("citation", "")).lower()
        for tok, expect in VENUE_TOKENS.items():
            if tok in k.lower() and expect not in cit:
                add("F3-naming", "WARN", k, f"key implies '{tok}' but citation lacks '{expect}'")
    # F1 dangling sources + F4 confidence + F7 mobility provenance
    low = 0
    for cat in ("absorbers", "etls", "htls"):
        for m, md in db.get(cat, {}).items():
            for pn, pd in md.get("parameters", {}).items():
                src = pd.get("source")
                if src and src not in refkeys:
                    add("F1-integrity", "ERROR", f"{m}.{pn}", f"source '{src}' not in references")
                meas = pd.get("measured")
                if isinstance(meas, dict) and meas.get("source") not in refkeys:
                    add("F1-integrity", "ERROR", f"{m}.{pn}", f"measured.source '{meas.get('source')}' missing")
                if pd.get("confidence") not in CONF:
                    add("F4-confidence", "ERROR", f"{m}.{pn}", f"bad confidence '{pd.get('confidence')}'")
                if pd.get("confidence") == "LOW":
                    low += 1
                    if not pd.get("low_conf_reason") and not pd.get("notes"):
                        add("F4-confidence", "ERROR", f"{m}.{pn}", "LOW without reason/notes")
                if pd.get("role") == "device_effective":
                    if not isinstance(meas, dict):
                        add("F7-provenance", "ERROR", f"{m}.{pn}", "device_effective without measured block")
                    elif meas.get("source") == src:
                        add("F7-provenance", "ERROR", f"{m}.{pn}", "device value attributed to its own measurement")
    add("F4-confidence", "INFO", "database", f"{low} LOW-confidence parameters")
    return db, findings


if __name__ == "__main__":
    import sys
    db, fs = audit()
    errs = [f for f in fs if f["severity"] == "ERROR"]
    warns = [f for f in fs if f["severity"] == "WARN"]
    nref = len(db["_references"]); npar = sum(len(md.get("parameters", {}))
            for c in ("absorbers","etls","htls") for md in db[c].values())
    print(f"PROVENANCE AUDIT: {nref} refs, {npar} params -> {len(errs)} ERROR, {len(warns)} WARN")
    for f in fs:
        if f["severity"] != "INFO":
            print(f"  [{f['severity']}] {f['flag']} @ {f['where']}: {f['message']}")
    print("RESULT:", "FAIL" if errs else "PASS")
    sys.exit(1 if errs else 0)
