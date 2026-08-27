#!/usr/bin/env bash
set -euo pipefail

# One-shot install of the laparoscopic gym wiring into .venv-lap-gym:
# SRT-H (+ lapgym/test extras, which include env-ssl-wrapper) and, optionally,
# the real SurRoL PyBullet simulator (--with-surrol). Ends with a smoke test
# of tests/test_lap_gym.py so you know the install works.
#
#   ./setup_lap_gym.sh                # mock stand-in (any platform)
#   ./setup_lap_gym.sh --with-surrol  # real SurRoL simulator (patched build)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv-lap-gym"
PYTHON="${PYTHON:-python3.10}"

echo "[setup_lap_gym] creating venv: $VENV (python: $PYTHON)"
uv venv "$VENV" --python "$PYTHON"

echo "[setup_lap_gym] installing SRT-H + lapgym extras + env-ssl-wrapper"
uv pip install --python "$VENV/bin/python" -e "$ROOT[lapgym,test]"

if [[ "${1:-}" == "--with-surrol" ]]; then
    echo "[setup_lap_gym] installing real SurRoL simulator"
    CACHE="$ROOT/.lap-gym"
    mkdir -p "$CACHE"

    if [[ "$(uname)" == "Darwin" ]]; then
        echo "[setup_lap_gym] building patched pybullet (no macOS wheels upstream)"
        SRC="$CACHE/pybullet-src"
        mkdir -p "$SRC"
        python3 -m pip download pybullet --no-deps --no-binary :all: -d "$SRC" > /dev/null
        tar xzf "$SRC"/pybullet-*.tar.gz -C "$SRC"
        PB_DIR="$(find "$SRC" -maxdepth 1 -type d -name 'pybullet-[0-9]*' | head -1)"
        python3 - "$PB_DIR/examples/ThirdPartyLibs/zlib/zutil.h" <<'EOF'
import sys
path = sys.argv[1]
with open(path) as f:
    content = f.read()
old = '#if defined(MACOS) || defined(TARGET_OS_MAC)'
assert content.count(old) == 1
content = content.replace(old, '#if 0 /* patched: TARGET_OS_MAC breaks stdio.h on modern macOS */')
with open(path, 'w') as f:
    f.write(content)
EOF
        MACOSX_DEPLOYMENT_TARGET=13.0 uv pip install --python "$VENV/bin/python" "$PB_DIR"
    else
        echo "[setup_lap_gym] installing pybullet (wheel)"
        uv pip install --python "$VENV/bin/python" pybullet
    fi

    echo "[setup_lap_gym] installing patched SurRoL (editable)"
    SURROL="$CACHE/SurRoL"
    if [[ ! -d "$SURROL/.git" ]]; then
        git clone --quiet --depth 1 --branch main https://github.com/med-air/SurRoL.git "$SURROL"
    fi
    for d in tasks robots utils data; do
        touch "$SURROL/surrol/$d/__init__.py"
    done
    python3 - "$SURROL/surrol/gym/surrol_env.py" <<'EOF'
import sys
path = sys.argv[1]
with open(path) as f:
    content = f.read()
old = "                egl = pkgutil.get_loader('eglRenderer')\n                plugin = p.loadPlugin(egl.get_filename(), \"_eglRendererPlugin\")"
new = "                egl = pkgutil.get_loader('eglRenderer')\n                if egl is not None:\n                    try:\n                        p.loadPlugin(egl.get_filename(), \"_eglRendererPlugin\")\n                    except Exception:\n                        pass"
assert old in content
with open(path, 'w') as f:
    f.write(content.replace(old, new))
EOF
    uv pip install --python "$VENV/bin/python" "gym==0.26.2" "numpy<2"
    uv pip install --python "$VENV/bin/python" -e "$SURROL"
fi

echo "[setup_lap_gym] smoke test: tests/test_lap_gym.py"
"$VENV/bin/python" -m pytest "$ROOT/tests/test_lap_gym.py" -q

echo
echo "[setup_lap_gym] done. run:"
echo "    $VENV/bin/python laparoscopic_env.py train"
echo "    $VENV/bin/python -m pytest tests/test_lap_gym.py"
