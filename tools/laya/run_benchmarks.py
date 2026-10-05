"""Run the full design matrix serially to avoid competing inference jobs.

Each report is flushed after completed cases. A missing complete record is an
unfinished experiment, not evidence of success. Existing results are preserved.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    def python(script, *args):
        subprocess.run([sys.executable, str(here / script), *map(str, args)], check=True)
    for family in a.checkpoints:
        bundle = a.bundles / family
        plan, rust, reference, summary = [a.output / (family + suffix) for suffix in
                                         ("-plan.json", "-rust.jsonl", "-python.jsonl", "-summary.json")]
        print(f"Starting {family}: full 12-case matrix, 5 warmups + 100 measured calls each", flush=True)
        python("make_benchmark_plan.py", "--bundle", bundle, "--output", plan)
        subprocess.run([str(a.runner.resolve()), str(bundle), str(a.runtime), str(plan), str(rust)], check=True)
        python("benchmark_python.py", "--bundle", bundle, "--rust", rust, "--output", reference)
        python("compare_benchmarks.py", "--rust", rust, "--python", reference, "--output", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundles", "runtime", "runner", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--checkpoints", nargs="+", choices=["english", "multilingual", "typed-decisions"],
                   default=["english", "multilingual", "typed-decisions"])
    main(p.parse_args())
