from __future__ import annotations
import json
from pathlib import Path
import sys
from pamssw.standalone.paper_reference import load_ssw_checkpoint

base = Path(sys.argv[1])
for mode in ("mc", "uniform", "pam"):
    folder = base / mode
    summary = json.loads((folder / "summary.json").read_text())
    wrapper = json.loads((folder / "search-result.json").read_text())
    result = wrapper["result"]
    assert summary["status"] == "completed", (mode, summary["status"])
    assert result["status"] == "completed", (mode, result["status"])
    records = result["records"]
    assert result["initial"]["atoms"]
    assert len(records) == 2, (mode, len(records))
    assert wrapper["checkpoint_file"] == "checkpoint.pkl"
    checkpoint = load_ssw_checkpoint(folder / wrapper["checkpoint_file"])
    assert checkpoint.ls is not None and checkpoint.frozen is not None, (mode, "Native-LS inactive")
    landings = [record for record in records if record["landing"] is not None]
    assert landings, (mode, "no returned landing")
    assert summary["returned_landings"] == len(landings)
    assert result["evaluation_requests"] == result["initial"]["evaluation_requests"] + sum(
        row["evaluation_requests"] for row in records
    )
    fresh = json.loads((folder / "fresh-checks.json").read_text())
    assert fresh["candidate_count"] == 1 + len(landings)
    assert len(fresh["checks"]) == fresh["candidate_count"]
    assert fresh["fresh_requests"] == len(fresh["checks"])
    assert all(row["status"] == "fresh_completed" for row in fresh["checks"])
    assert fresh["checks"][0]["source"] == "initial" and fresh["checks"][0]["certified"]
    for ledger_name, expected in (("search-ledger.jsonl", summary["search_requests"]),
                                  ("fresh-ledger.jsonl", summary["fresh_requests"])):
        ledger = [json.loads(line) for line in (folder / ledger_name).read_text().splitlines()]
        paid = sum(row.get("charged", False) for row in ledger)
        assert paid == expected, (mode, ledger_name, paid, expected)
    if mode != "mc":
        pool = json.loads((folder / "pool-report.json").read_text())
        assert summary["adapter_call_count"] > 0
        assert pool["decisions"]
    print(mode, "records=2", "landings=", len(landings),
          "search=", summary["search_requests"], "fresh=", summary["fresh_requests"],
          "selector_calls=", summary["adapter_call_count"])
