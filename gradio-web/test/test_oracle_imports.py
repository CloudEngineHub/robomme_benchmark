from __future__ import annotations

from pathlib import Path


def test_oracle_logic_imports_without_historybench(reload_module):
    oracle_logic = reload_module("oracle_logic")
    assert oracle_logic is not None

    module_path = Path(oracle_logic.__file__).resolve()
    source = module_path.read_text(encoding="utf-8")
    assert "historybench" not in source


def test_oracle_logic_exports_builder_and_vqa(reload_module):
    oracle_logic = reload_module("oracle_logic")
    assert hasattr(oracle_logic, "BenchmarkEnvBuilder")
    assert hasattr(oracle_logic, "get_vqa_options")
