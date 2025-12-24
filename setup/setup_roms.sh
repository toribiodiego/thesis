#!/usr/bin/env bash
# Setup Atari 2600 ROMs for ALE using AutoROM
# Downloads legally-redistributable ROMs required for Gymnasium ALE environments

set -e

echo "Installing Atari 2600 ROMs via AutoROM..."
python -m AutoROM --accept-license

echo ""
echo "ROM installation complete!"
echo "Verify installation with: python -c 'import ale_py; print(ale_py.roms.list())'"
