#!/usr/bin/env bash
set -euo pipefail

# compute_version.sh — thin wrapper.
#
# The logic (VERSION.txt -> REF_NAME -> RUN_NUMBER, plus the four-component
# MSIX form, plus the GITHUB_ENV/GITHUB_OUTPUT writes) lives in ContainerHub:
# every consumer with a CI lane needs exactly this, and it was reimplemented
# here before. Expects REF_NAME and RUN_NUMBER in the environment; writes
# VERSION and MSIX_VERSION for subsequent steps.

version_util="ExternalLib/Kataglyphis-ContainerHub/linux/scripts/02-toolchain/rust/version_util.sh"

if [[ ! -f "$version_util" ]]; then
  echo "ERROR: ${version_util} not found." >&2
  echo "       The submodule is not checked out: git submodule update --init --recursive" >&2
  exit 1
fi

bash "$version_util" --github-env "VERSION.txt"
