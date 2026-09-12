"""Input identity survives relocation and refuses missing input files."""
import hashlib

import pytest

from pawcerto.artifacts import file_identity


def test_identity_tracks_bytes_across_chunks_and_relocation(tmp_path):
    data = b'robot-asset\x00' * 100000
    first, second = tmp_path / 'asset', tmp_path / 'moved'
    first.write_bytes(data)
    first_id = file_identity(first)
    first.rename(second)
    second_id = file_identity(second)
    assert first_id['sha256'] == second_id['sha256'] == hashlib.sha256(data).hexdigest()
    assert second_id['path'] == str(second.resolve())
    with pytest.raises(FileNotFoundError):
        file_identity(first)
