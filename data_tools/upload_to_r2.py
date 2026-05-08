"""
Upload landmark JSONs from data_tools/landmarks/ to a Cloudflare R2 bucket.

R2 exposes an S3-compatible API, so we use boto3.

----------------------------------------------------------------------------
Setup
----------------------------------------------------------------------------
1. In the Cloudflare dashboard:
     - Create an R2 bucket (e.g. `isl-landmarks`).
     - Create an R2 API token with Object Read & Write on that bucket.
     - On the bucket's Settings tab, enable a public r2.dev URL OR connect a
       custom domain. Note the public hostname; that's what you'll set on
       Vercel as VITE_LANDMARKS_BASE_URL.

2. Create a `.env` next to this script (it's git-ignored) or export these
   environment variables:
       R2_ACCOUNT_ID=...
       R2_ACCESS_KEY_ID=...
       R2_SECRET_ACCESS_KEY=...
       R2_BUCKET=isl-landmarks

3. Install deps once:
       pip install boto3 python-dotenv

4. Run:
       python data_tools/upload_to_r2.py            # upload all new/changed
       python data_tools/upload_to_r2.py --force    # re-upload everything
       python data_tools/upload_to_r2.py --dry-run  # show what would happen

----------------------------------------------------------------------------
Note: the four files Hello.json, Wait.json, Problem.json, Please.json are NOT
uploaded -- they ship with the frontend bundle (frontend/public/landmarks/)
so the initial UI never depends on R2.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


LANDMARKS_DIR = Path(__file__).parent / "landmarks"

# These 4 are bundled with the frontend (frontend/public/landmarks/) and must
# not exist in R2 -- keeping them out of the bucket guarantees the bundled
# copies are the only source of truth.
EXCLUDE = {"Hello.json", "Wait.json", "Problem.json", "Please.json"}


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def make_client():
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5}),
    )


def already_uploaded(s3, bucket: str, key: str, local_md5: str) -> bool:
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise
    # ETag is the MD5 for single-part PUTs (which our small JSONs always are).
    return head.get("ETag", "").strip('"') == local_md5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-upload even if R2 already has a matching ETag.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be uploaded but don't upload.")
    args = parser.parse_args()

    if not LANDMARKS_DIR.exists():
        print(f"error: {LANDMARKS_DIR} does not exist", file=sys.stderr)
        return 1

    bucket = os.environ.get("R2_BUCKET", "isl-landmarks")
    s3 = None if args.dry_run else make_client()

    files = sorted(LANDMARKS_DIR.glob("*.json"))
    if not files:
        print(f"No .json files in {LANDMARKS_DIR}.")
        return 0

    uploaded = skipped = excluded = 0
    for f in files:
        if f.name in EXCLUDE:
            print(f"skip (bundled with frontend): {f.name}")
            excluded += 1
            continue

        local_md5 = md5sum(f)
        if not args.force and not args.dry_run and already_uploaded(s3, bucket, f.name, local_md5):
            print(f"skip (unchanged):              {f.name}")
            skipped += 1
            continue

        size_mb = f.stat().st_size / 1_048_576
        print(f"upload  ({size_mb:6.2f} MB):           {f.name}")
        if args.dry_run:
            uploaded += 1
            continue

        s3.upload_file(
            str(f),
            bucket,
            f.name,
            ExtraArgs={
                "ContentType": "application/json",
                # JSONs are content-addressed by name -- safe to cache aggressively.
                "CacheControl": "public, max-age=31536000, immutable",
            },
        )
        uploaded += 1

    print(
        f"\nDone. uploaded={uploaded}  skipped_unchanged={skipped}  "
        f"skipped_bundled={excluded}  bucket={bucket}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
