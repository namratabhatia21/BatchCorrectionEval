"""Download the three benchmark datasets used in this project.

The data are distributed inside the docker image
``jinmiaochenlab/batch-effect-removal-benchmarking`` (Tran et al., Genome Biology 2020).
This script streams the image's single filesystem layer from Docker Hub (about 3.8 GB) and
extracts only the folders that are needed, so neither docker nor 3.8 GB of disk is required:

    batch_effect/dataset2  -> Mouse Cell Atlas
    batch_effect/dataset4  -> Human Pancreas
    batch_effect/dataset7  -> Mouse Retina

Usage:
    python download_data.py [--out data]
"""

import argparse
import gzip
import json
import os
import shutil
import tarfile
import urllib.error
import urllib.request

REPO = "jinmiaochenlab/batch-effect-removal-benchmarking"
FOLDERS = ("dataset2", "dataset4", "dataset7")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _get(url, token=None, accept=None):
    """GET with the registry token. Redirects (blobs are served from a CDN with a
    pre-signed URL) are followed without the Authorization header."""
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if accept:
        req.add_header("Accept", accept)
    try:
        return urllib.request.build_opener(_NoRedirect).open(req)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            return urllib.request.urlopen(e.headers["Location"])
        if e.code == 429:
            raise SystemExit("Docker Hub rate limit reached for anonymous pulls; "
                             "try again later (the limit resets within a few hours).")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    token = json.load(_get(
        f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{REPO}:pull"
    ))["token"]
    manifest = json.load(_get(
        f"https://registry-1.docker.io/v2/{REPO}/manifests/latest", token,
        "application/vnd.docker.distribution.manifest.v2+json",
    ))
    layer = max(manifest["layers"], key=lambda l: l["size"])["digest"]
    print(f"Streaming layer {layer} ...")
    os.makedirs(args.out, exist_ok=True)
    with _get(f"https://registry-1.docker.io/v2/{REPO}/blobs/{layer}", token) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                parts = member.name.split("/")
                if len(parts) < 3 or parts[0] != "batch_effect" or parts[1] not in FOLDERS:
                    continue
                if not member.isfile():
                    continue
                dest = os.path.join(args.out, parts[1], parts[-1])
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                src = tar.extractfile(member)
                if dest.endswith(".gz"):
                    dest = dest[:-3]
                    src = gzip.GzipFile(fileobj=src)
                with open(dest, "wb") as f:
                    shutil.copyfileobj(src, f)
                print(f"  {dest}")
    print("Done.")


if __name__ == "__main__":
    main()
