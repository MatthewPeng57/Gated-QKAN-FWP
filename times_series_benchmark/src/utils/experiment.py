# Copyright 2026 Matthew Peng and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from datetime import datetime
import json
import os
import subprocess
import sys
import platform

# ===============================
# Path Construction
# ===============================

class Tee:
	def __init__(self, *streams):
		self.streams = streams
	def write(self, data):
		for s in self.streams:
			s.write(data)
			s.flush()
	def flush(self):
		for s in self.streams:
			s.flush()

def build_result_path(args, experiment_root: Path):

	timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S_%f")
	run_name = f"RUN_{timestamp}"

	MODEL_EXTRA_PARAMS = {
		"qqkanfwp": [],
		"lqkanfwp": [],
		"qkanlfwp": [],
		"qkanvfwp": [],
	}

	if args.model not in MODEL_EXTRA_PARAMS:
		raise ValueError("args.model incompatible")

	parts = [
		f"DATASET_{args.dataset}",
		f"MODEL_{args.model}",
	]

	# model-specific parameters
	for prefix, attr in MODEL_EXTRA_PARAMS[args.model]:
		value = getattr(args, attr)
		parts.append(f"{prefix}_{value}")

	# common parameters
	parts += [
		f"HIDDEN_SIZE_{args.hidden_size}",
		f"QNN_DEPTH_{args.qnn_depth}",
		f"SEQ_LEN_{args.window_len}",
		f"SEED_{args.seed}",
	]

	run_path = Path(*parts)

	save_dir = Path(args.save_dir)

	# --- Safety checks ---

	if any(part == ".." for part in run_path.parts):
		raise ValueError("run_path must not contain '..'")

	if any(part == ".." for part in save_dir.parts):
		raise ValueError("--save_dir must not contain '..'")

	if run_path.is_absolute():
		raise ValueError("run_path must not be an absolute path")

	if save_dir.is_absolute():
		raise ValueError("--save_dir must be a relative path (e.g. 'results'), not an absolute path")

	result_path = (experiment_root / save_dir / run_path / run_name).resolve()

	# Ensure the resolved path stays inside the experiment root
	result_path.relative_to(experiment_root)
	result_path.mkdir(parents=True, exist_ok=True)


	return result_path
# ===============================
# Metadata Saving
# ===============================

def save_args_json(args, result_path: Path):
	with open(result_path / "args.json", "w", encoding="utf-8") as f:
		json.dump(vars(args), f, indent=4, sort_keys=True)


def save_git_revision(result_path: Path):
	try:
		commit = subprocess.check_output(
			["git", "rev-parse", "HEAD"],
			stderr=subprocess.DEVNULL
		).decode().strip()

		branch = subprocess.check_output(
			["git", "rev-parse", "--abbrev-ref", "HEAD"],
			stderr=subprocess.DEVNULL
		).decode().strip()

		with open(result_path / "git_revision.txt", "w") as f:
			f.write(f"branch: {branch}\n")
			f.write(f"commit: {commit}\n")

	except Exception:
		with open(result_path / "git_revision.txt", "w") as f:
			f.write("Git information not available.\n")


def save_environment_snapshot(result_path: Path):
	"""
	Writes:
	- environment.yaml      (conda env export, when conda is available)
	- conda_list.txt        (conda list)
	- requirements.txt      (pip freeze)
	- python_info.txt       (python -V + executable)
	"""
	result_path.mkdir(parents=True, exist_ok=True)

	# 1) Python info (always available)
	with open(result_path / "python_info.txt", "w") as f:
		f.write(f"python_version: {sys.version}\n")
		f.write(f"python_executable: {sys.executable}\n")

	# 2) pip freeze (always worth saving; a conda export can miss pip-only packages)
	try:
		with open(result_path / "requirements.txt", "w") as f:
			subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f, check=True)
	except Exception as e:
		with open(result_path / "pip_freeze_error.txt", "w") as f:
			f.write(repr(e) + "\n")

	# 3) conda-specific artifacts (only when conda is available)
	#    Note: this block is skipped automatically outside a conda environment.
	try:
		# On some systems conda is not on PATH, but usually it is inside a conda env.
		subprocess.run(["conda", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

		# conda env export (includes the pip: section; still keep requirements.txt too)
		with open(result_path / "environment.yaml", "w") as f:
			subprocess.run(["conda", "env", "export"], stdout=f, stderr=subprocess.DEVNULL, check=True)

		# conda list (easy to diff)
		with open(result_path / "conda_list.txt", "w") as f:
			subprocess.run(["conda", "list"], stdout=f, stderr=subprocess.DEVNULL, check=True)

		# 4) Stricter: explicit spec (reproduces the env exactly on the same platform,
		#    but is pinned to that platform and may not work cross-platform).
		with open(result_path / "conda_explicit.txt", "w") as f:
			subprocess.run(["conda", "list", "--explicit"], stdout=f, stderr=subprocess.DEVNULL, check=True)

		# conda env name (handy for the record)
		env_name = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
		with open(result_path / "conda_env_name.txt", "w") as f:
			f.write(env_name + "\n")

	except Exception:
		# Not in conda, or conda unavailable -> nothing to do.
		pass


def generate_experiment_readme(args, result_path: Path) -> None:

	result_path = Path(result_path)
	readme_path = result_path / "README.md"

	command_line = " ".join(sys.argv)

	try:
		git_commit = subprocess.check_output(
			["git", "rev-parse", "HEAD"],
			stderr=subprocess.DEVNULL
		).decode().strip()
	except Exception:
		git_commit = "N/A"

	try:
		git_branch = subprocess.check_output(
			["git", "rev-parse", "--abbrev-ref", "HEAD"],
			stderr=subprocess.DEVNULL
		).decode().strip()
	except Exception:
		git_branch = "N/A"

	args_dict = vars(args)

	content = f"""
	# Experiment Run

	## Basic Info
	- Timestamp: {datetime.now().isoformat(timespec="seconds")}
	- Dataset: {getattr(args, "dataset", "N/A")}
	- Model: {getattr(args, "model", "N/A")}
	- Seed: {getattr(args, "seed", "N/A")}

	---

	## Command Used

		{command_line}

	---

	## Git Info
	- Branch: {git_branch}
	- Commit: {git_commit}

	---

	## Full Arguments

		{json.dumps(args_dict, indent=4)}

	---

	## System Info
	- Python Version: {sys.version}
	- Python Executable: {sys.executable}
	- Platform: {platform.platform()}

	---

	## Notes
	This folder is a self-contained experiment artifact.
	To reproduce:
	1. Checkout the git commit above.
	2. Restore the environment.
	3. Run the command listed above.
	"""

	readme_path.write_text(content.strip() + "\n", encoding="utf-8")
