from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from embedding_utils import project_path


REPO_ZIP_URL = "https://github.com/HIT-SCIR/plm-nlp-code/archive/refs/heads/main.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the HIT-SCIR Word2Vec training code used by the assignment.")
    parser.add_argument("--out-dir", default=str(project_path("training_code")))
    parser.add_argument("--zip-path", default=str(project_path("plm-nlp-code-main.zip")))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    zip_path = Path(args.zip_path)

    print(f"Downloading {REPO_ZIP_URL} ...")
    urllib.request.urlretrieve(REPO_ZIP_URL, zip_path)

    extract_dir = project_path("_plm_nlp_code_extract")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    candidates = list(extract_dir.glob("*/chp5"))
    if not candidates:
        raise SystemExit("Could not find chp5 in the downloaded repository.")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(candidates[0], out_dir)
    shutil.rmtree(extract_dir)

    print(f"Training code is ready in {out_dir}.")
    print("Run cbow.py and skipgram.py there, then copy cbow.vec and skipgram.vec to embeddings/.")


if __name__ == "__main__":
    main()
