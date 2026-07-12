"""Launch one ``generate.py`` process per TFRecord shard with GNU parallel.

Example:

    python script/main_generate.py \
      --model_path checkpoint.ckpt \
      --waymo_path /data/waymo/validation \
      --out_dir generated \
      --jobs 4 \
      -- --video

Everything after ``--`` is forwarded to ``generate.py``.
"""

import argparse
import glob
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def discover_tfrecords(path):
    """Return all TFRecord shard files behind a directory, glob, file, or @ spec."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.tfrecord-*-of-*")))
        if not files:
            files = sorted(glob.glob(os.path.join(path, "*.tfrecord*")))
    elif "@" in os.path.basename(path):
        prefix, count = path.rsplit("@", 1)
        count = int(count)
        files = [f"{prefix}-{i:05d}-of-{count:05d}" for i in range(count)]
    elif glob.has_magic(path):
        files = sorted(glob.glob(path))
    else:
        files = [path]

    if not files:
        raise FileNotFoundError(f"No TFRecord files found for {path}")
    missing = [file for file in files if not os.path.isfile(file)]
    if missing:
        raise FileNotFoundError(f"Missing TFRecord shard: {missing[0]}")
    return files


def build_command(args, shard, extra_args):
    return [
        args.python,
        args.generate_script,
        "--model_path", args.model_path,
        "--waymo_path", shard,
        "--out_dir", args.out_dir,
        "--device", args.device,
        "--num_scenes", str(args.num_scenes_per_shard),
        *extra_args,
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Launch generate.py for TFRecord shards using GNU parallel")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--waymo_path", required=True)
    parser.add_argument("--out_dir", default="generated_scenes")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--num_scenes_per_shard", type=int, default=-1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--generate_script",
        default=str(Path(__file__).with_name("generate.py")))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("generate_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if not os.path.isfile(args.generate_script):
        parser.error(f"generate.py not found: {args.generate_script}")

    extra_args = list(args.generate_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    shards = discover_tfrecords(args.waymo_path)
    commands = [build_command(args, shard, extra_args) for shard in shards]
    command_text = "\n".join(shlex.join(command) for command in commands) + "\n"

    print(
        f"Found {len(shards)} TFRecord shard(s); launching {args.jobs} at a time",
        flush=True,
    )
    if args.dry_run:
        print(command_text, end="")
        return

    parallel = shutil.which("parallel")
    if parallel is None:
        raise FileNotFoundError("GNU parallel is not installed or not in PATH")

    os.makedirs(args.out_dir, exist_ok=True)
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(
        [
            parallel,
            "--jobs", str(args.jobs),
            "--ungroup",
            "--halt", "now,fail=1",
        ],
        input=command_text,
        text=True,
        check=True,
        env=child_env,
    )


if __name__ == "__main__":
    main()
