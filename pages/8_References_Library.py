"""References library: every citation in the tool, searchable, with DOI."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import streamlit as st

st.set_page_config(page_title="References", page_icon="📚", layout="wide")
st.title("📚 Reference Library")
st.caption(
    "Every material parameter, model equation and benchmark target in this tool "
    "cites a peer-reviewed source with a DOI. This page aggregates the perovskite/CdTe "
    "material-database references (materials) and the final-release multi-technology registry. "
    "'Verified' means the citation was cross-checked against the Consensus academic "
    "index and/or publisher metadata during the audited build; unverified entries "
    "carry canonical, widely-cited DOIs pending the next audit pass "
    "(scripts/verify_references.py)."
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rows = []

# final-release multi-technology registry
mt = json.load(open(os.path.join(ROOT, "data", "multi_technology_database.json")))
for key, ref in mt["references"].items():
    rows.append({
        "Key": key, "Registry": "multi-technology",
        "Citation": ref["citation"], "DOI": ref["doi"],
        "Used for": ref.get("used_for", ""),
        "Confidence": ref.get("confidence", ""),
        "Verified": "✅" if ref.get("verified") else "—",
    })

# materials material database references
mat = json.load(open(os.path.join(ROOT, "data", "materials_database.json")))
for key, ref in mat.get("_references", {}).items():
    if isinstance(ref, dict):
        rows.append({
            "Key": key, "Registry": "materials database (materials)",
            "Citation": ref.get("citation", ref.get("full", str(ref))),
            "DOI": ref.get("doi", ""),
            "Used for": ref.get("used_for", "material parameters"),
            "Confidence": ref.get("confidence", ""),
            "Verified": "✅" if ref.get("verified") else "—",
        })

df = pd.DataFrame(rows)
q = st.text_input("🔎 Search citations / DOIs / usage", "")
if q:
    mask = df.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(),
                    axis=1)
    df = df[mask]

st.dataframe(
    df, use_container_width=True, height=560,
    column_config={"DOI": st.column_config.TextColumn("DOI")})

n_ver = (pd.DataFrame(rows)["Verified"] == "✅").sum()
st.metric("References in registry", len(rows),
          f"{n_ver} verified in audited builds")

st.markdown(
    "**Resolve a DOI**: prepend `https://doi.org/` to any DOI string above. "
    "**Export**: the full registries live in `data/multi_technology_database.json` "
    "and `data/materials_database.json` (machine-readable, per-parameter provenance)."
)
st.download_button("⬇️ Export reference table (CSV)",
                   pd.DataFrame(rows).to_csv(index=False),
                   "references.csv")
