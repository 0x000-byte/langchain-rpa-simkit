# tests/test_simple_chain.py
from langchain_rpa_simkit.simple_chain import plan, plan_batch

def test_plan_returns_steps():
    out = plan("extract invoices from email")
    assert "Plan for:" in out
    # Should contain numbered steps
    assert any(k in out for k in ["1)", "1.", "1-"])

def test_plan_batch_multiple():
    tasks = ["collect PDFs", "parse tables"]
    outs = plan_batch(tasks)
    assert len(outs) == 2
    assert "collect PDFs" in outs[0]
    assert "parse tables" in outs[1]

