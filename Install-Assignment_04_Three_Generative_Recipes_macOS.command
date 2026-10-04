#!/usr/bin/env bash
#=============================================================================
#region SCRIPT DETAILS
#=============================================================================
#.SYNOPSIS
#Checks for macOS, verifies prerequisites, and installs the required Python
#packages for Assignment 4 (Three Generative Recipes) into a local .venv.
#
#.DESCRIPTION
#- Verifies Python 3.10+ is installed
#- Creates (or reuses) a project-local virtual environment in .venv
#  (avoids Homebrew's "externally-managed-environment" pip error)
#- Upgrades pip inside the venv
#- Runs env_setup.py, which detects imports from the assignment script,
#  installs any missing packages, and reports the compute device
#- Offers a quick smoke test or the full run after setup
#
#.NOTES
#Double-click in Finder, or run:  bash Install-Assignment_04_Three_Generative_Recipes_macOS.command
#If macOS says it cannot be opened: right-click > Open, or run  chmod +x <this file>  once.
#Unattended:  LAUNCH=none bash Install-Assignment_04_Three_Generative_Recipes_macOS.command
#Written for the bash 3.2 that ships with macOS (no associative arrays).
#=============================================================================
#endregion
#=============================================================================
#region Prerequisites
#=============================================================================
#  OS CHECK: Only run on macOS (Darwin)
OS_NAME=$(uname)
if [ "$OS_NAME" != "Darwin" ]; then
    echo "OS is not macOS. This script is only intended for macOS devices."
    exit 666
fi
#=============================================================================
#endregion
#=============================================================================
#region FUNCTIONS
#=============================================================================

set -eo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m';  GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m';     RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; }
header()  { echo -e "\n${BOLD}${CYAN}$*${RESET}\n"; }
pause_exit() {
    if [ -z "${LAUNCH:-}" ]; then read -rp "Press Enter to exit..." _; fi
    exit "${1:-1}"
}

# ── Change to the script's own directory ─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ASSIGNMENT_SCRIPT="scripts/Assignment_04_Three_Generative_Recipes.py"
ENV_SETUP="env_setup.py"
VENV_DIR=".venv"
VENV_PY="$VENV_DIR/bin/python"
MIN_MAJOR=3
MIN_MINOR=10

[ -z "${LAUNCH:-}" ] && clear
echo -e "${BOLD}=================================================================${RESET}"
echo -e "${BOLD}   Assignment 4 Three Generative Recipes — macOS Installer       ${RESET}"
echo -e "${BOLD}=================================================================${RESET}"
echo ""
#=============================================================================
#endregion
#=============================================================================
#region EXECUTION
#=============================================================================
#Step 1: Find Python 3.10+
#=============================================================================
header "Step 1: Locating Python ${MIN_MAJOR}.${MIN_MINOR}+"

PYTHON=""
for cmd in python3 python3.14 python3.13 python3.12 python3.11 python3.10 python; do
    if command -v "$cmd" >/dev/null 2>&1 && \
       "$cmd" -c "import sys; sys.exit(0 if sys.version_info >= ($MIN_MAJOR, $MIN_MINOR) else 1)" 2>/dev/null; then
        PYTHON="$cmd"
        PY_VER=$("$PYTHON" -c "import sys; print(sys.version.split()[0])")
        success "Found: $PYTHON  (Python $PY_VER)"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python ${MIN_MAJOR}.${MIN_MINOR}+ was not found on this system."
    echo ""
    echo -e "  ${BOLD}Install Python from one of these sources:${RESET}"
    echo "    • Official:  https://www.python.org/downloads/"
    echo "    • Homebrew:  brew install python"
    echo ""
    echo "  After installing Python, re-run this script."
    pause_exit 1
fi

if [ "$(uname -m)" = "x86_64" ]; then
    warn "Intel Mac detected. Recent PyTorch releases ship macOS wheels for Apple Silicon only,"
    warn "so the torch install may fail here. If it does, use the Colab notebook instead (see COLAB.md)."
fi

#=============================================================================
#Step 2: Check application files
#=============================================================================
header "Step 2: Checking application files"

for f in "$ASSIGNMENT_SCRIPT" "$ENV_SETUP"; do
    if [ -f "$f" ]; then
        success "Found: $f"
    else
        error "Not found: $f"
        warn "Run this installer from the project folder (next to env_setup.py)."
        pause_exit 1
    fi
done

#=============================================================================
#Step 3: Create / reuse the virtual environment
#=============================================================================
header "Step 3: Preparing virtual environment (.venv)"

if [ -x "$VENV_PY" ]; then
    success "Existing .venv found - reusing it"
else
    "$PYTHON" -m venv "$VENV_DIR" || { error "Failed to create .venv"; pause_exit 1; }
    success "Created .venv"
fi

#=============================================================================
#Step 4: Upgrade pip
#=============================================================================
header "Step 4: Ensuring pip is up to date"

"$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
if ! "$VENV_PY" -m pip install --upgrade pip --quiet; then
    error "pip upgrade failed."
    warn "With python.org Python, an SSL error usually means you need to run"
    warn "'Install Certificates.command' from /Applications/Python 3.x/ once."
    pause_exit 1
fi
success "pip is ready"

#=============================================================================
#Step 5: Detect & install required packages (env_setup.py parses the assignment imports)
#=============================================================================
header "Step 5: Detecting & installing packages"
info "This can take a few minutes the first time: torch is large."

if ! "$VENV_PY" "$ENV_SETUP"; then
    error "env_setup.py failed - see the messages above."
    pause_exit 1
fi

#=============================================================================
#Step 6: Verify imports
#=============================================================================
header "Step 6: Verifying imports"

VERIFY_FAILED=0
for import_name in torch numpy pandas matplotlib; do
    if "$VENV_PY" -c "import $import_name" >/dev/null 2>&1; then
        success "$import_name successfully imported"
    else
        error "Failed to import: $import_name"
        VERIFY_FAILED=1
    fi
done

if [ "$VERIFY_FAILED" -eq 1 ]; then
    echo ""
    error "One or more packages failed verification."
    pause_exit 1
fi

info "Macs have no NVIDIA GPU, so the local run uses the CPU (Apple MPS is detected but not used:"
info "the Burgers solver needs float64). The graded run is on a Colab T4 - see COLAB.md."

#=============================================================================
#Step 7: Launch prompt
#=============================================================================
echo ""
echo -e "${BOLD}=================================================================${RESET}"
echo -e "${BOLD}   Setup Complete — Assignment 4 Three Generative Recipes       ✓${RESET}"
echo -e "${BOLD}=================================================================${RESET}"
echo ""

CHOICE="${LAUNCH:-}"
if [ -z "$CHOICE" ]; then
    echo -e "  ${BOLD}1${RESET}  Quick smoke test (~2-3 min, short training - NOT submission numbers)"
    echo -e "  ${BOLD}2${RESET}  Full run (~15 min on CPU; writes results/*_cpu.*)"
    echo -e "  ${BOLD}3${RESET}  Exit without running"
    echo ""
    while true; do
        read -rp "  Enter your choice [1/2/3]: " c
        case "$c" in
            1) CHOICE=quick; break ;;
            2) CHOICE=full;  break ;;
            3) CHOICE=none;  break ;;
            *) warn "Please enter 1, 2 or 3." ;;
        esac
    done
fi

case "$CHOICE" in
    quick)
        info "Running quick smoke test ..."
        HALDEN_QUICK=1 "$VENV_PY" "$ASSIGNMENT_SCRIPT"
        ;;
    full)
        info "Running full pipeline ..."
        "$VENV_PY" "$ASSIGNMENT_SCRIPT"
        ;;
    *)
        info "To run manually:"
        echo "    .venv/bin/python $ASSIGNMENT_SCRIPT"
        info "Quick smoke test:"
        echo "    HALDEN_QUICK=1 .venv/bin/python $ASSIGNMENT_SCRIPT"
        ;;
esac

echo ""
#=============================================================================
#endregion
#=============================================================================
