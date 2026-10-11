"""The streaming JSON reader must give exactly what json.load gives, whatever the chunk size."""
import json

from studio.soccertrack import iter_json_array


def test_iter_json_array_matches_json_load(tmp_path):
    data = {"info": {"a": 1}, "images": [{"x": 1}],
            "annotations": [{"id": i, "nested": {"s": "a, ] {" * (i % 3)}, "v": [i, i * 2]} for i in range(200)]}
    path = tmp_path / "gsr.json"
    path.write_text(json.dumps(data, indent=4))
    for chunk in (7, 64, 1000, 1 << 20):
        assert list(iter_json_array(path, "annotations", chunk_size=chunk)) == data["annotations"]


def test_iter_json_array_empty(tmp_path):
    path = tmp_path / "gsr.json"
    path.write_text('{"annotations": [ ]}')
    assert list(iter_json_array(path, "annotations", chunk_size=3)) == []
