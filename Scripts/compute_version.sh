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

# Four-component MSIX version. The rule (append .0 to a three-part version,
# otherwise fall back to 0.1.0.0) is ContainerHub's `version_util.sh
# --normalize`; this used to be a second implementation of it here, which is
# how the two get to disagree. Only the VERSION.txt sourcing and the
# GITHUB_ENV/GITHUB_OUTPUT plumbing below are genuinely this repo's.
#
# Checked against the old inline rule over 11 inputs (verified 2026-08-07):
# identical on every one except a leading "v" (`v2.3.4` -> 2.3.4.0 there,
# 0.1.0.0 here). That case cannot arise: ${ver} has already had a leading "v"
# stripped above, and the `^[0-9]` guard has already replaced anything else
# with RUN_NUMBER, so ${ver} always starts with a digit by this point.
version_util="ExternalLib/Kataglyphis-ContainerHub/linux/scripts/02-toolchain/rust/version_util.sh"
if [[ -x "$version_util" || -f "$version_util" ]]; then
  msix_ver="$(bash "$version_util" --normalize "$ver")"
else
  # Submodule not checked out (a shallow consumer clone, or a local run before
  # `git submodule update`). Keep the lane alive with the same rule inline
  # rather than failing the build over a missing helper.
  echo "WARNING: ${version_util} not found; using the inline MSIX fallback." >&2
  msix_ver="$ver"
  [[ "$msix_ver" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] && msix_ver="${msix_ver}.0"
  [[ "$msix_ver" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || msix_ver="0.1.0.0"
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
