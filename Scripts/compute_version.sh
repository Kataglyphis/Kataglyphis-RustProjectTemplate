#!/usr/bin/env bash
set -euo pipefail

# compute_version.sh
# Expects REF_NAME and RUN_NUMBER in environment (passed from workflow expressions).
# Writes VERSION and MSIX_VERSION to GITHUB_ENV and prints them to stdout.


REF_NAME=${REF_NAME:-}
RUN_NUMBER=${RUN_NUMBER:-}

# Primary source: VERSION.txt at repository root. Read first non-empty line if present.
version_file="VERSION.txt"
file_ver=""
if [[ -f "$version_file" ]]; then
  # read first non-empty line
  while IFS= read -r line; do
    line_trimmed="$(echo "$line" | tr -d '\r' | sed -e 's/^\s*//' -e 's/\s*$//')"
    if [[ -n "$line_trimmed" ]]; then
      file_ver="$line_trimmed"
      break
    fi
  done < "$version_file"
fi

# Start with file version if available, else ref name (stripped of leading v), else run number
if [[ -n "$file_ver" ]]; then
  ver="${file_ver#v}"
else
  ver="${REF_NAME#v}"
  if [[ -z "$ver" ]]; then
    ver="$RUN_NUMBER"
  fi
fi

# If version doesn't start with a digit, fall back to RUN_NUMBER
if [[ ! "$ver" =~ ^[0-9] ]]; then
  ver="$RUN_NUMBER"
fi

# Compute MSIX version: ensure four numeric components (major.minor.patch.build)
msix_ver="$ver"
if [[ "$msix_ver" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  msix_ver="${msix_ver}.0"
fi
if [[ ! "$msix_ver" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  # If file provided a non-numeric MSIX-like version, don't propagate; default to 0.1.0.0
  msix_ver="0.1.0.0"
fi

# Export results for subsequent steps
echo "VERSION=$ver" >> "${GITHUB_ENV:-/dev/null}"
echo "MSIX_VERSION=$msix_ver" >> "${GITHUB_ENV:-/dev/null}"

echo "Computed VERSION=$ver"
echo "Computed MSIX_VERSION=$msix_ver"

# Also add to GITHUB_OUTPUT if available (for step outputs)
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "VERSION=$ver" >> "$GITHUB_OUTPUT"
  echo "MSIX_VERSION=$msix_ver" >> "$GITHUB_OUTPUT"
fi

exit 0
