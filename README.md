<p align="center">
  <img src="https://img.shields.io/badge/rl--cli-Malware%20Analysis%20CLI-blue?style=for-the-badge" alt="rl-cli">
</p>

<h1 align="center">ReversingLabs CLI Client</h1>

<p align="center">
  <strong>Modular command-line client for ReversingLabs TitaniumCloud and A1000 malware analysis platforms</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.13%20%7C%203.14-blue?style=flat-square&logo=python&logoColor=white" alt="Python Versions"></a>
  <a href="https://github.com/seifreed/ReversingLabs-Client/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"></a>
  <a href="https://github.com/seifreed/ReversingLabs-Client"><img src="https://img.shields.io/badge/A1000%20SDK%20methods-54%20of%2067-brightgreen?style=flat-square" alt="A1000 SDK Coverage"></a>
</p>

<p align="center">
  <a href="https://github.com/seifreed/ReversingLabs-Client/stargazers"><img src="https://img.shields.io/github/stars/seifreed/ReversingLabs-Client?style=flat-square" alt="GitHub Stars"></a>
  <a href="https://github.com/seifreed/ReversingLabs-Client/issues"><img src="https://img.shields.io/github/issues/seifreed/ReversingLabs-Client?style=flat-square" alt="GitHub Issues"></a>
  <a href="https://buymeacoffee.com/seifreed"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-support-yellow?style=flat-square&logo=buy-me-a-coffee&logoColor=white" alt="Buy Me a Coffee"></a>
</p>

---

## Overview

**rl-cli** is a Python CLI and library that provides seamless access to ReversingLabs TitaniumCloud and A1000 malware analysis platforms, with Rich-formatted terminal output and full automation-friendly JSON/YAML modes.

### Key Features

| Feature | Description |
|---------|-------------|
| **Malware Analysis** | Submit files for deep static and dynamic analysis |
| **Multi-AV Scanning** | Detection results from 40+ antivirus engines |
| **Network Intelligence** | URL, domain, and IP reputation and reports |
| **Sample Management** | Upload, download, delete, reanalyze, and batch operations |
| **YARA Integration** | Rulesets, online repositories, retro scans (cloud and local) |
| **Classification & Tags** | Set and manage custom threat classifications |
| **Broad A1000 SDK Coverage** | Wraps 54 of the 67 ReversingLabs A1000 SDK methods. Of the rest, four submit-and-report helpers are re-implemented here with better progress reporting, `configuration_dump` is a local string formatter replaced by `config-dump`, `list_extracted_files_v2_aggregated` pages for a list one request already returns, the two YARA sync-time methods set a timestamp no workflow reads, and the five `*_aggregated` walks are done here instead because the SDK's loops exit only when a page says it is the last — an appliance answering an empty page that promises another never stops them |
| **Rich Terminal UI** | Formatted output with colors, tables, and panels |
| **CLI + Library** | Use as command-line tool or Python package |

### Supported Outputs

```text
Terminal        Rich-formatted tables, panels, and colors
Automation      JSON, YAML, table, raw
LLM pipelines   TOON (Token-Oriented Object Notation)
Code scanning   SARIF 2.1.0 (GitHub Code Scanning compatible)
Reports         JSON, TitaniumCore, PDF, HTML (dynamic analysis)
```

---

## Installation

### From Source

```bash
git clone https://github.com/seifreed/ReversingLabs-Client.git
cd ReversingLabs-Client
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

### Development Extras

```bash
pip install -e ".[dev]"
```

---

## Quick Start

Create a configuration file at `~/.rl-cli.yaml` (sections live under a profile key):

```yaml
default:
  titanium_cloud:
    username: "your_ticloud_username"
    password: "your_ticloud_password"

  a1000:
    host: "https://your-a1000-instance.local"
    token: "your_a1000_token_here"
```

Or use environment variables:

```bash
export TICLOUD_USERNAME="your_username"
export TICLOUD_PASSWORD="your_password"
export A1000_HOST="https://your-a1000.local"
export A1000_TOKEN="your_token"
```

The same variables can go in a `.env` file in the working directory instead
of being exported. Either way they take precedence over the config file, so
an exported `A1000_TOKEN` overrides the token in `config.yaml` while the file
still supplies everything the environment leaves unset.

Then:

```bash
# Test API connections
rl-cli a1000 test
rl-cli config check-access

# Check file reputation by hash
rl-cli ticloud reputation SHA256_HASH

# Upload and analyze a file
rl-cli a1000 upload /path/to/file.exe --wait

# Get detailed report
rl-cli a1000 report SHA256_HASH
```

---

## Usage

### Command Line Interface

```bash
# JSON output for automation. Status messages go to stderr, so stdout is
# the document alone; a failed command exits non-zero.
rl-cli --output json a1000 report <HASH> | jq '.results[0].classification'

# Token-efficient output for LLM pipelines
rl-cli --output toon a1000 report <HASH>

# SARIF 2.1.0 log for code-scanning ingestion
rl-cli --output sarif ticloud reputation <HASH> > results.sarif

# Batch operations
rl-cli a1000 batch-reanalyze -h <HASH1> -h <HASH2>

# YARA hunting
rl-cli a1000 yara-create suspicious_strings rules.yar
rl-cli a1000 yara-matches suspicious_strings            # every match
rl-cli a1000 yara-matches suspicious_strings <HASH>     # just this sample

# Network intelligence
rl-cli a1000 domain-report <DOMAIN>
rl-cli a1000 ip-report <IP>
```

### Available Options (Main Commands)

| Command | Description |
|---------|-------------|
| `rl-cli ticloud` | TitaniumCloud: file, URL, domain and IP reputation, AV scanners, analysis, search, sample download |
| `rl-cli a1000` | A1000: upload, reports, samples, YARA, classification, network intel |
| `rl-cli config` | Configuration: `show`, `init`, `save`, `create-profile`, `list-profiles`, `check-access` |

A failed command exits non-zero, with one deliberate exception: `config
check-access` is a diagnostic, and it exits 0 whenever it managed to probe and
report — including when it reports that nothing is reachable. It is the command
the CLI tells you to run when a service is unreachable, so what it measured is
its answer rather than its failure, and under `set -e` a diagnostic that exits 1
exactly when it has something to say aborts the script before you can read it. A
non-zero exit from it means the check could not be made at all (an unusable
config file, for instance). Branch on the measurement itself, which is on
stdout:

```bash
rl-cli --output json config check-access | jq -e '.summary.services_available > 0'
```

### Global Flags

| Option | Description |
|--------|-------------|
| `--config <file>`, `-c` | Path to configuration file |
| `--profile <name>`, `-p` | Configuration profile to use |
| `--output <fmt>`, `-o` | Output format: `rich`, `json`, `yaml`, `table`, `raw`, `toon`, `sarif` |
| `--quiet`, `-q` | Suppress progress output; warnings and errors remain |
| `--verbose`, `-v` | Verbose output |
| `--version` | Print the version and exit |

### TitaniumCloud Commands

| Command | Description |
|---------|-------------|
| `reputation` | Malware presence and classification for one hash, or a batch in one request |
| `analysis` | Full analysis record for a hash |
| `av-scanners` | Per-engine AV scanner results for a hash |
| `analyze-url` | Reputation for a URL |
| `search` | Advanced Search over the TitaniumCloud corpus |
| `upload` | Submit a file for TitaniumCloud analysis |
| `download` | Write the stored sample to disk (live malware) |
| `download-status` | Whether TitaniumCloud holds the sample bytes for a hash |
| `domain-report` | Threat intelligence for a domain |
| `domain-files` | Files downloaded from a domain |
| `domain-urls` | URLs seen on a domain |
| `domain-ips` | Domain-to-IP resolutions |
| `domain-related` | Domains sharing a top parent domain |
| `ip-report` | Threat intelligence for an IP address |
| `ip-files` | Files downloaded from an IP address |
| `ip-urls` | URLs seen on an IP address |
| `ip-domains` | IP-to-domain resolutions |
| `url-files` | Files downloaded from a URL |
| `uri-index` | SHA-1s of every sample seen at a URI, domain, IPv4 or email address |

`reputation` takes several hashes — as arguments, repeated `-h`, or `-f` with
one per line — and grades them in a single bulk request instead of one metered
request per hash. A batch must be all of one hash type, which is what the bulk
endpoint queries by. A hash named twice is looked up once, and one hash or five
hundred are reported the same way: one graded record per sample, in every output
format.

The nine lookups from `domain-files` down return the first page of results. The
--all flag pages through every result, which on a busy address is the whole
corpus at 1000 records a metered page; --max-results N stops after N results and
pages on its own, so --all need not be given as well. A walk that comes back
exactly N results long says that more may be waiting, since what ended it may
have been the cap rather than the corpus.

`download` writes live malware: the file lands owner-only (0600), a symlink at
the destination is refused rather than followed, and an interrupted download
leaves nothing behind.

```bash
# Bulk triage: one request, not 500
rl-cli --output json ticloud reputation -f hashes.txt

# Pivot from a C2 address
rl-cli ticloud ip-report <IP>
rl-cli ticloud ip-files <IP> --max-results 500
rl-cli ticloud domain-related <DOMAIN>

# Fetch the sample itself
rl-cli ticloud download-status <HASH>
rl-cli ticloud download <HASH> --output-dir ./samples
```

### A1000 Command Groups (52 Commands)

| Group | Commands |
|-------|----------|
| File operations | `upload`, `upload-and-analyze`, `status`, `report`, `summary-report`, `titanium-report` |
| Sample management | `list`, `search`, `download`, `delete`, `reanalyze`, `batch-delete`, `batch-reanalyze` |
| Extracted files | `extracted`, `containers` |
| Classification & tags | `set-classification`, `get-classification`, `delete-classification`, `add-tags`, `get-tags`, `remove-tags` |
| YARA rulesets | `yara-list`, `yara-create`, `yara-content`, `yara-delete`, `yara-toggle`, `yara-matches`, `yara-publish`, `yara-update-now`, `yara-update-interval` |
| YARA retro hunts | `yara-cloud-retro`, `yara-cloud-retro-status`, `yara-local-retro`, `yara-local-retro-status` |
| YARA repositories | `yara-repo-list`, `yara-repo-create`, `yara-repo-update`, `yara-repo-delete` |
| Network intelligence | `domain-report`, `ip-report`, `ip-files`, `ip-domains`, `ip-urls`, `network-url-report` |
| URL analysis | `submit-url`, `url-report`, `url-status` |
| Reports | `dynamic-report-create`, `dynamic-report-status`, `dynamic-report`, `report --format pdf` |
| Connection | `test`, `config-dump` |

`ip-files`, `ip-domains` and `ip-urls` return the first page of results; add
`--all` to page through every result. `extracted` lists every extracted file in
one request, so it takes no paging flag; its `-a/--download-all` writes the
extracted files to disk.

Eight commands ask before they act, because what they do cannot be undone:
`delete`, `batch-delete`, `delete-classification`, `remove-tags` with no `-t`,
`yara-delete`, `yara-cloud-retro -o clear`, `yara-repo-delete`, and
`config create-profile`, which overwrites a stored profile's credentials. Each
takes `--yes`, which takes the confirmation as given — a script or a cron job
has no answer to type, and an unanswered prompt aborts and exits 1 having done
nothing. The prompt is still the default, so a bare invocation at a terminal
asks as it always did.

```bash
# Scripted removal: nothing on stdin to answer with
rl-cli a1000 yara-repo-delete 7 --yes
rl-cli a1000 batch-delete -f hashes.txt --yes
```

`yara-repo-create` and `yara-repo-update` take the credential the appliance uses
to clone a private rule repository — a GitHub PAT, typically. Pass
`--api-token-stdin` instead of `--api-token <token>`: an argument is readable in
ps output by every other user on the machine for as long as the call runs, and
the line stays in your shell history file afterwards, which is why the bare flag
is discouraged. With `--api-token-stdin` the token is piped in, or typed at a
hidden prompt when stdin is a terminal. Exactly one trailing newline is stripped
and nothing else, so a piped token arrives as written. Supplying both ways at
once is a usage error rather than a precedence rule. The flag itself still works
unchanged — a public repository is stated as an empty `--api-token`, or by
passing neither to `yara-repo-create`.

```bash
# Interactive: nothing is echoed, nothing is left in the history file
rl-cli a1000 yara-repo-create --url <REPO_URL> --name org-rules --api-token-stdin

# Scripted: the token comes from a secret store, never from argv
pass show github/rules-pat | rl-cli a1000 yara-repo-create --url <REPO_URL> --name org-rules --api-token-stdin
```

---

## Python Library

### Basic Usage

Each A1000 service covers one area of the appliance, and an `A1000Session` is
one connection to it: build the services you need from a single session and
they authenticate once between them, however many areas you touch. The
TitaniumCloud side splits the same way, without a session to share — there is
no connection to share, since each call builds its own API handle — so
enriching an address takes `TitaniumCloudNetworkService` and the file
endpoints take `TitaniumCloudService`.

```python
from rl_cli.services import (
    A1000ReportService,
    A1000SampleService,
    A1000Session,
    TitaniumCloudNetworkService,
    TitaniumCloudService,
    upload_and_get_report,
)
from rl_cli.config import get_settings
from pathlib import Path

# `get_settings()`, not `Settings()`: the constructor only holds the values it
# is handed, so a bare `Settings()` talks to ReversingLabs' public clouds
# whatever your config file says. This is the discovery the CLI itself does.
settings = get_settings()
ticloud = TitaniumCloudService(settings)

# Check file reputation
reputation = ticloud.get_file_reputation("SHA256_HASH")

# Enriching an address needs the network half and nothing else. Every pivot
# comes in two: the plain call answers the first page, the `_aggregated` one
# walks every page and takes a `max_results` budget.
network = TitaniumCloudNetworkService(settings)
resolutions = network.get_domains_from_ip("8.8.8.8")

# One session, one connection: both services below share it.
session = A1000Session(settings)
samples = session.service(A1000SampleService)
reports = session.service(A1000ReportService)

# Upload and analyze a file. `task_id` is the digest A1000 returned for the
# submission — its SHA256 when the response carries one, otherwise the SHA1 —
# so the same value polls the analysis and fetches the finished report. It is
# absent when the upload failed (`None`) and when the appliance took the file
# but its answer could not be read, so ask for it rather than indexing.
result = samples.upload_file(Path("/path/to/file.exe"), comment="Automated analysis")
task_id = (result or {}).get("task_id")
if task_id and samples.wait_for_analysis(task_id, timeout=300):
    report = reports.get_report(task_id)

# Or as one step: the workflow spans both areas, so it takes both services.
summary = upload_and_get_report(samples, reports, Path("/path/to/file.exe"), "summary")
```

### YARA Hunting

```python
from rl_cli.services import A1000YaraService

# A third area of the appliance, still over the connection opened above.
yara = session.service(A1000YaraService)
sample_hashes = [task_id] if task_id else []

# get_yara_matches returns the list of matching samples (empty when none),
# or None if the call failed.
matches = {
    h: entries
    for h in sample_hashes
    if (entries := yara.get_yara_matches("suspicious_strings", h))
}

# Ends the connection for every service built from the session.
session.close()
```

---

## Configuration

Configuration sources in priority order: CLI options → environment variables → config file → defaults.

Config file locations searched (in order): `./config.yaml`, `./.rl-cli.yaml`, `~/.config/rl-cli/config.yaml`, `~/.rl-cli.yaml`.

See [config.example.yaml](config.example.yaml) and [.env.example](.env.example) for the complete reference, including proxy support, SSL verification, and output settings.

---

## Security

- Never commit credentials — use config files (`chmod 600`) or environment variables
- Enable SSL verification in production; prefer API tokens over username/password
- Never pass the YARA repository token as `--api-token <token>`: argv is world-readable in `ps` and the line is kept in your shell history. `yara-repo-create` and `yara-repo-update` take `--api-token-stdin`, which pipes it in or prompts for it without echoing. It is deliberately not read from the environment or `.env` — that file is plaintext in the working directory, and nothing here creates it 0600 or keeps it out of a commit
- The commands that delete something prompt first, and `--yes` is the only thing that skips the prompt: neither `--quiet` nor a closed stdin is taken for a yes, so a non-interactive run has to say that it means it
- `download` and `extracted --download-all` print a warning naming the destination before writing anything. No classification is consulted: every file A1000 hands back is treated as live malware. `--output-dir` defaults to the current directory, so point it at an isolated directory or sandbox
- Codebase scanned with **Bandit** and **pip-audit**, linted with **Ruff**

---

## Requirements

- Python 3.13 or 3.14
- ReversingLabs API credentials (TitaniumCloud and/or A1000)
- See [pyproject.toml](pyproject.toml) for dependencies and extras. Every
  runtime dependency is declared with both a floor and a major-version
  ceiling, so a fresh install resolves the same majors this project is
  tested against
- The package ships a PEP 561 `py.typed` marker, so when it is imported as a
  library its annotations are type information your own mypy will read,
  rather than `Any`

---

## Contributing

Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) has the gate
commands, the places adding a command or an output format touches, and where
the conventions are written down.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Enable the pre-push hook once per clone — it is not on by default, so on a
fresh clone nothing checks a push until CI does:

```bash
git config core.hooksPath .githooks
```

It runs [pre-push-check.sh](pre-push-check.sh), which scans for
credentials and malware samples and then runs the same quality gate CI
runs. To run that gate by hand:

```bash
ruff check rl_cli/ tests/
ruff format --check rl_cli/ tests/
mypy rl_cli/ tests/
bandit -r rl_cli/ -q
pip-audit --skip-editable
pytest -q
```

`pytest -q` enforces the 95% line-and-branch coverage floor on its own: the
flags live in `pyproject.toml`, so no spelling of the whole-suite command
skips it. A narrower run — one file, a nodeid, `-k`, `-m`, `--deselect` —
still prints the coverage report but is not held to the floor, because 95%
of `rl_cli` was never that run's claim.

---

## Support the Project

If this project is useful in your workflows, you can support development:

<a href="https://buymeacoffee.com/seifreed" target="_blank">
  <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50">
</a>

---

## Acknowledgments

- Built on top of the official [ReversingLabs SDK for Python](https://github.com/reversinglabs/reversinglabs-sdk-py3)
- Uses [Rich](https://github.com/Textualize/rich), [Click](https://github.com/pallets/click), and [Pydantic](https://github.com/pydantic/pydantic)

---

## License

This project is licensed under the MIT license. See [LICENSE](LICENSE).

**Attribution**
- Author: **Marc Rivero López** | [@seifreed](https://github.com/seifreed)
- Repository: [github.com/seifreed/ReversingLabs-Client](https://github.com/seifreed/ReversingLabs-Client)

---

<p align="center">
  <sub>Built for practical malware analysis and threat intelligence automation</sub>
</p>
