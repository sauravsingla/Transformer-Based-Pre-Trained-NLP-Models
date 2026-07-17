# Security Policy

## Supported code

Security fixes target the current `main` branch. Historical notebooks are preserved for research provenance and may use outdated dependencies; they should be run only in an isolated environment.

## Reporting a vulnerability

Do not disclose exploitable vulnerabilities, credentials, private datasets, access tokens, or personal information in a public issue.

Report the problem privately to the repository owner through the contact options on the GitHub profile. Include:

- A clear description of the issue.
- A minimal reproduction.
- Affected file and version or commit.
- Potential impact.
- Suggested mitigation, when available.

Allow reasonable time for investigation before public disclosure.

## Data and model safety

- Never commit the raw tweet dataset unless its licence explicitly permits redistribution.
- Never commit secrets, API keys, access tokens, private URLs, or user credentials.
- Treat downloaded model files and datasets as untrusted inputs.
- Review third-party checkpoint licences and model cards before use.
- Prefer pinned or bounded dependencies and run dependency checks regularly.
- Do not load untrusted Python pickle files or arbitrary checkpoints from unknown sources.

## Historical notebooks

The original notebooks date from the paper period and are retained to show the original workflow. Their package APIs and dependency assumptions may be obsolete. The maintained implementation under `src/` is the recommended execution path.
