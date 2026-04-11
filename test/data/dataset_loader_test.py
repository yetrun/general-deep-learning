from data.wiki.loader import doc_load
from env.resolve import resolve_path


def test_dataset_load_mini_c4():
    data_dir = resolve_path("data/dev/mini_c4")
    ds = doc_load(data_dir=data_dir, glob_pattern="*.txt", cycle_length=1)

    result = list(ds.as_numpy_iterator())
    assert len(result) == 10

    assert result[0] == b"first document of first file"
    assert result[1] == b"second document of first file"
    assert result[2] == b"third document of first file"
    assert result[3] == b"first document of second file"
    assert result[4] == b"second document of second file"
    assert result[5] == b"third document of second file"
    assert result[6] == b"fourth document of second file"
    assert result[7] == b"first document of third file"
    assert result[8] == b"second document of third file"
    assert result[9] == b"third document of third file"
