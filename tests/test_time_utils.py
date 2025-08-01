from unittest.mock import patch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import utils.utils as u


def reset_cache():
    u._TIME_CACHE = {"offset": 0, "fetched_at": 0, "failures": 0}


def test_remote_time_cached():
    reset_cache()
    with patch("utils.utils.requests.get") as mock_get, \
         patch("utils.utils.time.time", return_value=1) as mock_time:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"unixtime": 1000}
        assert u.current_utc_unixtime(force_sync=True) == 1000
        assert mock_get.call_count == 1

    with patch("utils.utils.time.time", return_value=100):
        assert u.current_utc_unixtime() == 1099
        assert mock_get.call_count == 1


def test_force_sync_triggers_remote_fetch():
    reset_cache()
    with patch("utils.utils.requests.get") as mock_get, \
         patch("utils.utils.time.time", return_value=1):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"unixtime": 1000}
        u.current_utc_unixtime(force_sync=True)
        assert mock_get.call_count == 1

        mock_get.return_value.json.return_value = {"unixtime": 1100}
        with patch("utils.utils.time.time", return_value=100):
            assert u.current_utc_unixtime(force_sync=True) == 1100
        assert mock_get.call_count == 2


def test_remote_failure_fallback():
    reset_cache()
    with patch("utils.utils.requests.get", side_effect=Exception("boom")), \
         patch("utils.utils.time.time", return_value=1):
        assert u.current_utc_unixtime(force_sync=True) == 1

    with patch("utils.utils.requests.get", side_effect=Exception("boom")), \
         patch("utils.utils.time.time", return_value=600):
        assert u.current_utc_unixtime(force_sync=True) == 600

    with patch("utils.utils.requests.get", side_effect=Exception("boom")), \
         patch("utils.utils.time.time", return_value=1200):
        assert u.current_utc_unixtime(force_sync=True) == 1200

    assert u._TIME_CACHE["failures"] >= 3
