"""Every code block in docs/API.md must actually run — executable docs."""
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_api_md_examples_execute():
    md = open(os.path.join(os.path.dirname(__file__), "..", "docs", "API.md")).read()
    blocks = re.findall(r"```python\n(.*?)```", md, re.S)
    assert len(blocks) >= 6
    ns = {}
    for b in blocks:
        exec(compile(b, "<API.md>", "exec"), ns)
