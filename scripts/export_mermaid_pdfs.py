from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "doc" / "docs"
OUTPUT_DIR = REPO_ROOT / "doc" / "mermaid-pdf"
PUPPETEER_CONFIG = REPO_ROOT / "scripts" / "mermaid-puppeteer.json"
DOCKERFILE_PATH = REPO_ROOT / "dockerfiles" / "mermaid-cli-pdf" / "Dockerfile"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
DOCKER_IMAGE = "options-pricing-mermaid-cli:latest"
CHROME_BIN = (
    "/usr/bin/chromium"
)
MERMAID_BLOCK_RE = re.compile(r"```mermaid\s*\r?\n(.*?)\r?\n```", re.DOTALL)


def iter_mermaid_blocks() -> list[dict[str, str | int]]:
    entries: list[dict[str, str | int]] = []
    for markdown_path in sorted(DOCS_DIR.rglob("*.md")):
        text = markdown_path.read_text(encoding="utf-8")
        matches = list(MERMAID_BLOCK_RE.finditer(text))
        if not matches:
            continue
        relative_md = markdown_path.relative_to(DOCS_DIR)
        md_output_dir = OUTPUT_DIR / relative_md.with_suffix("")
        md_output_dir.mkdir(parents=True, exist_ok=True)
        for index, match in enumerate(matches, start=1):
            source = match.group(1).strip() + "\n"
            entry = {
                "source_markdown": relative_md.as_posix(),
                "diagram_index": index,
                "diagram_type": source.splitlines()[0].strip(),
                "mmd_relpath": (relative_md.with_suffix("") / f"{index:02d}.mmd").as_posix(),
                "pdf_relpath": (relative_md.with_suffix("") / f"{index:02d}.pdf").as_posix(),
                "source": source,
            }
            entries.append(entry)
    return entries


def write_sources(entries: list[dict[str, str | int]]) -> None:
    for entry in entries:
        mmd_path = OUTPUT_DIR / str(entry["mmd_relpath"])
        mmd_path.parent.mkdir(parents=True, exist_ok=True)
        mmd_path.write_text(str(entry["source"]), encoding="utf-8")


def render_pdf(entry: dict[str, str | int]) -> None:
    mmd_path = OUTPUT_DIR / str(entry["mmd_relpath"])
    pdf_path = OUTPUT_DIR / str(entry["pdf_relpath"])
    repo_mount = f"{REPO_ROOT.resolve().as_posix()}:/work"
    input_in_container = f"/work/{mmd_path.relative_to(REPO_ROOT).as_posix()}"
    output_in_container = f"/work/{pdf_path.relative_to(REPO_ROOT).as_posix()}"
    puppeteer_in_container = f"/work/{PUPPETEER_CONFIG.relative_to(REPO_ROOT).as_posix()}"

    command = [
        "docker",
        "run",
        "--rm",
        "-e",
        "HOME=/home/mermaidcli",
        "-e",
        f"PUPPETEER_EXECUTABLE_PATH={CHROME_BIN}",
        "-v",
        repo_mount,
        DOCKER_IMAGE,
        "-i",
        input_in_container,
        "-o",
        output_in_container,
        "-p",
        puppeteer_in_container,
        "-f",
    ]
    subprocess.run(command, check=True)


def ensure_docker_image() -> None:
    inspect_command = ["docker", "image", "inspect", DOCKER_IMAGE]
    inspect_result = subprocess.run(
        inspect_command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if inspect_result.returncode == 0:
        return

    if not DOCKERFILE_PATH.exists():
        raise FileNotFoundError(f"No existe {DOCKERFILE_PATH}")

    build_command = [
        "docker",
        "build",
        "-f",
        str(DOCKERFILE_PATH),
        "-t",
        DOCKER_IMAGE,
        str(REPO_ROOT),
    ]
    subprocess.run(build_command, check=True)


def render_all(entries: list[dict[str, str | int]]) -> None:
    for position, entry in enumerate(entries, start=1):
        source_markdown = entry["source_markdown"]
        pdf_relpath = entry["pdf_relpath"]
        print(f"[{position}/{len(entries)}] {source_markdown} -> {pdf_relpath}")
        render_pdf(entry)


def write_manifest(entries: list[dict[str, str | int]]) -> None:
    manifest_entries = []
    for entry in entries:
        manifest_entries.append(
            {
                "source_markdown": entry["source_markdown"],
                "diagram_index": entry["diagram_index"],
                "diagram_type": entry["diagram_type"],
                "mmd_relpath": entry["mmd_relpath"],
                "pdf_relpath": entry["pdf_relpath"],
            }
        )
    MANIFEST_PATH.write_text(
        json.dumps(manifest_entries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if not DOCS_DIR.exists():
        print(f"No existe {DOCS_DIR}", file=sys.stderr)
        return 1
    if not PUPPETEER_CONFIG.exists():
        print(f"No existe {PUPPETEER_CONFIG}", file=sys.stderr)
        return 1

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = iter_mermaid_blocks()
    if not entries:
        print("No se han encontrado bloques Mermaid.")
        return 0

    ensure_docker_image()
    write_sources(entries)
    render_all(entries)
    write_manifest(entries)
    print(f"Exportados {len(entries)} diagramas a {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
