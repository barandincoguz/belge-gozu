import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_colab_notebook_reuses_checkout_and_requests_single_page_batches() -> None:
    notebook = json.loads(
        (REPO / "notebooks" / "build_dense_artifacts_colab.ipynb").read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert "git', 'fetch', 'origin'" in code
    assert "'--batch-size', '1'" in code
    assert "Yeni Colab oturumu kullanın" not in code
