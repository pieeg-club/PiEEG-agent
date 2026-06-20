#!/bin/sh
# Point git at the repo's tracked hooks directory. Run once after cloning.
set -e
ROOT="$(git rev-parse --show-toplevel)"
git -C "$ROOT" config core.hooksPath .githooks
chmod +x "$ROOT/.githooks/pre-commit" 2>/dev/null || true
echo "Installed git hooks (core.hooksPath = .githooks)."
