"""
Docker-based sandbox for executing coding agent commands and running test harnesses.
Each sandbox manages a single Docker container lifecycle.
"""

import docker
import json
import re
import time
import base64
import tarfile
import io
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


SANDBOX_IMAGE = "coding-sandbox:latest"
COMMAND_TIMEOUT = 30
TEST_TIMEOUT = 120


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int


class DockerSandbox:
    """Manages a single Docker container for code execution and testing."""

    def __init__(self, sandbox_id: str, image: str = SANDBOX_IMAGE):
        self.sandbox_id = sandbox_id
        self.image = image
        self.client = docker.from_env()
        self.container = None

    def create(self, context_files: Dict[str, str], harness_files: Dict[str, str] = None):
        """Create container and populate with initial files."""
        self.container = self.client.containers.run(
            self.image,
            detach=True,
            name=f"sandbox-{self.sandbox_id}",
            mem_limit="512m",
            cpu_period=100000,
            cpu_quota=100000,
            network_mode="none",
            working_dir="/code",
        )

        for filepath, content in context_files.items():
            full_path = f"/code/{filepath}"
            self._write_file(full_path, content)

        if harness_files:
            for filepath, content in harness_files.items():
                full_path = f"/code/harness/{filepath}"
                self._write_file(full_path, content)

    def _write_file(self, container_path: str, content: str):
        """Write a file into the container using tar archive."""
        dir_path = os.path.dirname(container_path)
        self.container.exec_run(f"mkdir -p {dir_path}")

        tar_stream = io.BytesIO()
        filename = os.path.basename(container_path)
        content_bytes = content.encode("utf-8")

        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            info = tarfile.TarInfo(name=filename)
            info.size = len(content_bytes)
            tar.addfile(info, io.BytesIO(content_bytes))

        tar_stream.seek(0)
        self.container.put_archive(dir_path, tar_stream)

    def exec(self, command: str, timeout: int = COMMAND_TIMEOUT) -> ExecResult:
        """Execute a bash command in the container."""
        if self.container is None:
            return ExecResult(stdout="", stderr="Error: sandbox not initialized", exit_code=1)

        try:
            exit_code, output = self.container.exec_run(
                ["bash", "-c", command],
                workdir="/code",
                demux=True,
            )
            stdout = output[0].decode("utf-8", errors="replace") if output[0] else ""
            stderr = output[1].decode("utf-8", errors="replace") if output[1] else ""

            max_len = 4096
            if len(stdout) > max_len:
                stdout = stdout[:max_len] + "\n... [truncated]"
            if len(stderr) > max_len:
                stderr = stderr[:max_len] + "\n... [truncated]"

            return ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code)
        except Exception as e:
            return ExecResult(stdout="", stderr=f"Execution error: {str(e)}", exit_code=1)

    def run_tests(self, harness: Dict[str, str]) -> Tuple[float, str]:
        """
        Run the CVDP test harness and return (pass_rate, raw_output).
        
        The harness contains:
        - src/.env: environment variables for test setup
        - src/test_runner.py: cocotb test runner
        - src/test_*.py: actual test cases
        """
        for filepath, content in harness.items():
            if filepath == "docker-compose.yml":
                continue
            full_path = f"/code/harness/{filepath}"
            self._write_file(full_path, content)

        env_content = harness.get("src/.env", "")
        env_vars = {}
        for line in env_content.strip().split("\n"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                env_vars[key.strip()] = val.strip()

        env_vars["PYTHONPATH"] = "/code/harness/src"

        env_str = " ".join(f'{k}="{v}"' for k, v in env_vars.items())
        test_cmd = f"cd /code/rundir && {env_str} pytest -o cache_dir=/code/rundir/.cache /code/harness/src/test_runner.py -v 2>&1"

        result = self.exec(test_cmd, timeout=TEST_TIMEOUT)
        raw_output = result.stdout + "\n" + result.stderr

        pass_rate = self._parse_pytest_output(raw_output)
        return pass_rate, raw_output

    def _parse_pytest_output(self, output: str) -> float:
        """Parse pytest output to extract pass rate."""
        # pytest summary line: "X passed, Y failed" or "X passed"
        passed_match = re.search(r"(\d+)\s+passed", output)
        failed_match = re.search(r"(\d+)\s+failed", output)
        error_match = re.search(r"(\d+)\s+error", output)

        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0
        errors = int(error_match.group(1)) if error_match else 0

        total = passed + failed + errors
        if total == 0:
            return 0.0
        return passed / total

    def destroy(self):
        """Remove the container."""
        if self.container:
            try:
                self.container.remove(force=True)
            except Exception:
                pass
            self.container = None

    def __del__(self):
        self.destroy()


class SandboxPool:
    """Manages a pool of Docker sandboxes for batch rollout."""

    def __init__(self, image: str = SANDBOX_IMAGE):
        self.image = image
        self.sandboxes: Dict[str, DockerSandbox] = {}
        self._counter = 0

    def create_batch(
        self,
        context_files_list: List[Dict[str, str]],
        harness_files_list: List[Dict[str, str]] = None,
        id_prefix: str = "batch",
    ) -> List[str]:
        """Create a batch of sandboxes, return list of sandbox IDs."""
        sandbox_ids = []
        for i, context_files in enumerate(context_files_list):
            sid = f"{id_prefix}_{self._counter}_{i}"
            self._counter += 1
            sandbox = DockerSandbox(sid, self.image)
            harness = harness_files_list[i] if harness_files_list else None
            sandbox.create(context_files, harness)
            self.sandboxes[sid] = sandbox
            sandbox_ids.append(sid)
        return sandbox_ids

    def exec_batch(self, sandbox_ids: List[str], commands: List[str]) -> List[ExecResult]:
        """Execute commands in multiple sandboxes."""
        results = []
        for sid, cmd in zip(sandbox_ids, commands):
            if sid in self.sandboxes and cmd:
                results.append(self.sandboxes[sid].exec(cmd))
            else:
                results.append(ExecResult(stdout="", stderr="Sandbox not found or empty command", exit_code=1))
        return results

    def run_tests_batch(
        self, sandbox_ids: List[str], harness_list: List[Dict[str, str]]
    ) -> List[Tuple[float, str]]:
        """Run tests in multiple sandboxes."""
        results = []
        for sid, harness in zip(sandbox_ids, harness_list):
            if sid in self.sandboxes:
                results.append(self.sandboxes[sid].run_tests(harness))
            else:
                results.append((0.0, "Sandbox not found"))
        return results

    def get_sandbox(self, sandbox_id: str) -> Optional[DockerSandbox]:
        return self.sandboxes.get(sandbox_id)

    def destroy_batch(self, sandbox_ids: List[str]):
        """Destroy a batch of sandboxes."""
        for sid in sandbox_ids:
            if sid in self.sandboxes:
                self.sandboxes[sid].destroy()
                del self.sandboxes[sid]

    def destroy_all(self):
        """Destroy all sandboxes."""
        for sandbox in self.sandboxes.values():
            sandbox.destroy()
        self.sandboxes.clear()

    def __del__(self):
        self.destroy_all()
