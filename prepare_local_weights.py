"""Download the official model and small BERT tokenizer/config files only."""
import json
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import BertConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parent
REVISION = "06bd674ecd2d18ea55ec0dd9183fa24603dfa7e1"
SHA256 = "94eabe4e2595dac1e04cc1703641458b13c1c624ce10072ba57dd96b625a41f5"


def main():
    destination = ROOT / "checkpoints"
    destination.mkdir(exist_ok=True)
    filename = Path(hf_hub_download("nikigoli/CountGDPlusPlus", "model.safetensors", revision=REVISION, local_dir=destination))
    digest = hashlib.sha256()
    with filename.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise RuntimeError("Downloaded model checksum does not match the pinned official weights")
    bert = destination / "bert-base-uncased"
    BertConfig.from_pretrained("google-bert/bert-base-uncased").save_pretrained(bert)
    AutoTokenizer.from_pretrained("google-bert/bert-base-uncased").save_pretrained(bert)
    (destination / "model-source.json").write_text(json.dumps(
        {"repo_id": "nikigoli/CountGDPlusPlus", "revision": REVISION, "sha256": SHA256,
         "bytes": filename.stat().st_size, "git_revision": "2ecbc4331d6730d017f52a4a97bbcbae7d5d714a"}, indent=2)+"\n")
    print(f"Weights ready in {destination}")


if __name__ == "__main__":
    main()
