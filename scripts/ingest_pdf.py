import sys
import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.ecoquery.pdf_ingest import ingest_pdf


def main():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    ingest_pdf(
        pdf_path=config["pdf_path"],
        output_dir=Path(config["vector_store_path"]).parent,
        chunk_size=config.get("chunk_size", 1000),
        chunk_overlap=config.get("chunk_overlap", 400),
    )


if __name__ == "__main__":
    main()
