"""
Fetch the DF40 CLIP checkpoint at container start-up (used by the Dockerfile).

Provide ONE of these environment variables:
  DF40_WEIGHTS_REPO   Hugging Face repo id that holds clip.pth, e.g. "yourname/deepshield-weights"
                      (set HF_TOKEN too if the repo is private; DF40_WEIGHTS_FILE overrides the filename)
  DF40_WEIGHTS_URL    Any direct-download URL for clip.pth

Does nothing if checkpoints/df40/clip.pth already exists.
"""

import os
import shutil
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.environ.get("DF40_CLIP_CHECKPOINT", os.path.join(ROOT, "checkpoints", "df40", "clip.pth"))


def main() -> int:
    if os.path.exists(DEST):
        print(f"[weights] Found {DEST}")
        return 0

    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    repo = os.environ.get("DF40_WEIGHTS_REPO")
    url = os.environ.get("DF40_WEIGHTS_URL")

    if repo:
        from huggingface_hub import hf_hub_download
        filename = os.environ.get("DF40_WEIGHTS_FILE", "clip.pth")
        print(f"[weights] Downloading {filename} from Hugging Face repo {repo} ...")
        path = hf_hub_download(repo_id=repo, filename=filename, token=os.environ.get("HF_TOKEN"))
        shutil.copyfile(path, DEST)
    elif url:
        print(f"[weights] Downloading from {url} ...")
        urllib.request.urlretrieve(url, DEST)
    else:
        print("[weights] ERROR: no checkpoint found and neither DF40_WEIGHTS_REPO nor "
              "DF40_WEIGHTS_URL is set. See checkpoints/README.md.", file=sys.stderr)
        return 1

    print(f"[weights] Saved to {DEST} ({os.path.getsize(DEST) / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
