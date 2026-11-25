import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "Final_LLM_Pipeline.ipynb"


def test_final_notebook_is_demo_only_and_cells_parse():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]

    assert notebook["nbformat"] == 4
    assert len(code_cells) == 3
    all_code = ["".join(cell["source"]) for cell in code_cells]
    assert "merge" not in "\n".join(all_code).lower()
    assert "quantiz" not in "\n".join(all_code).lower()
    runner = all_code[-1]
    assert re.search(r'DEMO_SCRIPT_REF = "[0-9a-f]{40}"', runner)
    assert "subprocess.run([sys.executable, str(DEMO_SCRIPT_PATH)], check=True)" in runner

    for source in all_code:
        if source.startswith("%%bash"):
            if shutil.which("bash") is None:
                pytest.skip("bash is unavailable for notebook shell-cell validation")
            result = subprocess.run(
                ["bash", "-n"],
                input=source.removeprefix("%%bash\n"),
                text=True,
                capture_output=True,
                check=False,
            )
            assert result.returncode == 0, result.stderr
        else:
            compile(source, str(NOTEBOOK), "exec")
