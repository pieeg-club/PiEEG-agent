@echo off
REM Point git at the repo's tracked hooks directory. Run once after cloning.
git config core.hooksPath .githooks
if %errorlevel% neq 0 (
    echo Failed to configure git hooks.
    exit /b %errorlevel%
)
echo Installed git hooks (core.hooksPath = .githooks).
