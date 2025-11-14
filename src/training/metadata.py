"""Metadata persistence for reproducibility (git info, config, seed).
"""

import os
import json
import subprocess
from typing import Optional, Dict, Any
import torch


def get_git_commit_hash() -> str:
    """
    Get current git commit hash.

    Returns:
        str: Git commit hash (short form), or 'unknown' if not in git repo
    """
    import subprocess

    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def get_git_status() -> dict:
    """
    Get detailed git repository status.

    Returns:
        dict: Git status information including commit, branch, dirty state
    """
    import subprocess

    status = {
        'commit_hash': 'unknown',
        'commit_hash_full': 'unknown',
        'branch': 'unknown',
        'dirty': False
    }

    try:
        # Get short commit hash
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        status['commit_hash'] = result.stdout.strip()

        # Get full commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        status['commit_hash_full'] = result.stdout.strip()

        # Get branch name
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        status['branch'] = result.stdout.strip()

        # Check if working tree is dirty
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True
        )
        status['dirty'] = len(result.stdout.strip()) > 0

    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return status


class MetadataWriter:
    """
    Writes reproducibility metadata for training runs.

    Persists configuration, seed, git commit, and other metadata to ensure
    experiments can be reproduced.

    Parameters
    ----------
    run_dir : str
        Directory to save metadata files

    Usage
    -----
    >>> writer = MetadataWriter(run_dir='runs/pong_123')
    >>> writer.write_metadata(config=config, seed=123, extra={'device': 'cuda'})
    """

    def __init__(self, run_dir: str):
        import os
        self.run_dir = run_dir

        # Create run directory
        os.makedirs(run_dir, exist_ok=True)

    def write_metadata(
        self,
        config: dict = None,
        seed: int = None,
        extra: dict = None,
        format: str = 'json'
    ):
        """
        Write complete metadata for reproducibility.

        Saves:
        - Configuration (if provided)
        - Random seed
        - Git commit hash and status
        - Timestamp
        - Python version
        - PyTorch version
        - Additional metadata

        Args:
            config: Configuration dictionary
            seed: Random seed
            extra: Additional metadata to include
            format: Output format ('json' or 'yaml')
        """
        import os
        import sys
        from datetime import datetime

        # Gather metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'git': get_git_status()
        }

        if seed is not None:
            metadata['seed'] = seed

        if config is not None:
            metadata['config'] = config

        if extra is not None:
            metadata.update(extra)

        # Write metadata
        if format == 'json':
            self._write_json(metadata)
        elif format == 'yaml':
            self._write_yaml(metadata)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'.")

        # Also write git info separately for easy access
        self._write_git_info(metadata['git'])

    def _write_json(self, metadata: dict):
        """Write metadata to JSON file."""
        import json
        import os

        json_path = os.path.join(self.run_dir, 'metadata.json')

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def _write_yaml(self, metadata: dict):
        """Write metadata to YAML file."""
        import os

        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML format. Install with: pip install pyyaml")

        yaml_path = os.path.join(self.run_dir, 'metadata.yaml')

        with open(yaml_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

    def _write_git_info(self, git_status: dict):
        """Write git information to separate file for easy reference."""
        import os

        git_info_path = os.path.join(self.run_dir, 'git_info.txt')

        with open(git_info_path, 'w') as f:
            f.write(f"Commit: {git_status['commit_hash']}\n")
            f.write(f"Full Hash: {git_status['commit_hash_full']}\n")
            f.write(f"Branch: {git_status['branch']}\n")
            f.write(f"Dirty: {git_status['dirty']}\n")

            if git_status['dirty']:
                f.write("\nWARNING: Working tree has uncommitted changes!\n")
                f.write("Results may not be fully reproducible.\n")

    def write_config(self, config: dict, format: str = 'yaml'):
        """
        Write configuration to separate file.

        Args:
            config: Configuration dictionary
            format: Output format ('json' or 'yaml')
        """
        import os
        import json

        if format == 'json':
            config_path = os.path.join(self.run_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        elif format == 'yaml':
            try:
                import yaml
            except ImportError:
                raise ImportError("PyYAML is required for YAML format. Install with: pip install pyyaml")

            config_path = os.path.join(self.run_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_metadata(self, format: str = 'json') -> dict:
        """
        Load metadata from file.

        Args:
            format: Format to load ('json' or 'yaml')

        Returns:
            dict: Loaded metadata
        """
        import os
        import json

        if format == 'json':
            metadata_path = os.path.join(self.run_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                return json.load(f)
        elif format == 'yaml':
            try:
                import yaml
            except ImportError:
                raise ImportError("PyYAML is required for YAML format. Install with: pip install pyyaml")

            metadata_path = os.path.join(self.run_dir, 'metadata.yaml')
            with open(metadata_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
