"""Prepare and smoke-test an installable fork wheel."""

import hashlib
import json
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile


def run(*args):
  subprocess.run(args, check=True)


root = pathlib.Path.cwd()
dist = root / "dist"
dist.mkdir(exist_ok=True)
wheel = next((root / "bazel-bin/python/litert_lm").glob(
    "litert_lm_api-0.17.0+quantdrift.1-*.whl"))
if sys.platform == "linux":
  # The upstream filename claims glibc 2.27 regardless of the build host.
  # Drop that tag before auditwheel assigns and verifies the actual baseline.
  run(sys.executable, "-m", "wheel", "tags", "--platform-tag",
      "linux_" + platform.machine(), str(wheel))
  wheel = next(wheel.parent.glob("*-linux_*.whl"))
  run(sys.executable, "-m", "auditwheel", "repair", "--plat",
      "manylinux_2_39_" + platform.machine(), "-w", str(dist), str(wheel))
else:
  shutil.copy2(wheel, dist / wheel.name)
wheel = next(dist.glob("*.whl"))
run(sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel))

# Import from site-packages so this checks the distributed native library.
import litert_lm  # pylint: disable=g-import-not-at-top

assert not pathlib.Path(litert_lm.__file__).is_relative_to(root)
with tempfile.TemporaryDirectory() as cache:
  with litert_lm.Engine(
      str(root / "runtime/testdata/test_lm.litertlm"),
      litert_lm.Backend.CPU(), max_num_tokens=32, cache_dir=cache,
  ) as engine:
    for prefix, middle, target in (
        ("Hello", " world", " again"),
        ("The", " sky", " is blue"),
        ("The capital of France", " is", " Paris"),
    ):
      middle_ids = engine.tokenize(middle)
      assert engine.tokenize(prefix + middle) == (
          engine.tokenize(prefix) + middle_ids)
      assert engine.tokenize(middle + target) == (
          middle_ids + engine.tokenize(target))
      with engine.create_session(apply_prompt_template=False) as session:
        session.run_prefill([prefix])
        expected = session.run_text_scoring([middle + target]).token_scores[0]
        expected = expected[len(middle_ids):]
      for context in ([prefix + middle], [prefix, middle]):
        with engine.create_session(apply_prompt_template=False) as session:
          for text in context:
            session.run_prefill([text])
          actual = session.run_text_scoring([target]).token_scores[0]
        assert len(actual) == len(expected) == len(engine.tokenize(target))
        assert all(abs(a - b) < 1e-5 for a, b in zip(actual, expected))

provenance = {
    "commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "platform": platform.platform(),
    "wheel": wheel.name,
    "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    "scoring_context_regression": "passed",
}
(dist / (wheel.name + ".json")).write_text(
    json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
print(json.dumps(provenance, indent=2))
