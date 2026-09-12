"""File identity shared by method-specific training and export records."""
import hashlib
from pathlib import Path


def file_identity(path):
    """Identify an existing file without loading a checkpoint or changing config."""
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest()}
