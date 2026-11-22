import json
from pathlib import Path
import unittest


class RecordsTest(unittest.TestCase):
    def test_known_input_contract(self):
        source = Path(__file__).resolve().parents[1] / "data" / "records.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        frames = {(item['group'], item['period']): item['population'] for item in payload['frames']}
        for item in payload['certificates'] + [payload['new_certificate']]:
            assert 0 <= item['lower'] <= item['upper'] <= frames[item['group'], item['period']]
            assert len(item['points']) == len(set(item['points']))
        assert any(item['event'] == '撤回凭证' for item in payload['history'])
        assert len({key for item in payload['requests'] for key in item['offsets']}) < len(payload['requests'])
