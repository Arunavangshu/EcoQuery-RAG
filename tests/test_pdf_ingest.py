from pathlib import Path


def test_placeholder():
    project_root = Path(__file__).resolve().parents[1]
    assert (project_root / "Apple_Environmental_Progress_Report_2024.pdf").exists()
