"""Dataset ingest, library include/load, and trainer attachment."""

from src.data.ingest import format_chat_turns, parse_path, parse_text
from src.hub import datasets as datasets_mod


def test_parse_blank_line_text():
    result = parse_text("hello world\n\nsecond example\n\nthird", filename="notes.txt", split="blank_line")
    assert result.sample_count == 3
    assert result.samples[1] == "second example"


def test_parse_chat_jsonl():
    raw = "\n".join(
        [
            '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello there"}]}',
            '{"prompt":"how are you","completion":"I am well."}',
        ]
    )
    result = parse_text(raw, filename="chat.jsonl", sector="chat")
    assert result.sample_count == 2
    assert "User: hi" in result.samples[0]
    assert "Assistant:\nhello there" in result.samples[0]
    assert "User: how are you" in result.samples[1]


def test_parse_alpaca_json():
    payload = '[{"instruction":"Translate","input":"hola","output":"hello"}]'
    result = parse_text(payload, filename="alpaca.json", sector="chat")
    assert result.sample_count == 1
    assert "User: Translate" in result.samples[0]
    assert "hola" in result.samples[0]
    assert "Assistant:\nhello" in result.samples[0]


def test_parse_csv_prompt_completion(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text("prompt,completion\n2+2,4\ncapital of France,Paris\n", encoding="utf-8")
    result = parse_path(path, sector="language")
    assert result.sample_count == 2
    assert "2+2" in result.samples[0]
    assert "4" in result.samples[0]


def test_format_chat_turns_skips_empty():
    text = format_chat_turns(
        [
            {"role": "system", "content": "Be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "hey"},
        ]
    )
    assert "System: Be brief" in text
    assert text.endswith("Assistant:\nhey")


def test_include_and_load_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_mod, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(datasets_mod, "DATASETS_REGISTRY_PATH", tmp_path / "hub_datasets.json")
    rec = datasets_mod.include_from_text(
        name="Support chats",
        text="User: hi\nAssistant:\nHello.\n\nUser: bye\nAssistant:\nGoodbye.",
        sector="chat",
        split="blank_line",
    )
    assert rec["sample_count"] == 2
    assert rec["id"]
    listed = datasets_mod.list_datasets()
    assert any(d["id"] == rec["id"] for d in listed)
    texts = datasets_mod.load_dataset_texts(rec["id"])
    assert len(texts) == 2
    assert "Hello." in texts[0]
    datasets_mod.mark_dataset_used(rec["id"], "Mini-Chat")
    stored = datasets_mod.get_dataset(rec["id"])
    assert "Mini-Chat" in stored["used_by"]
    assert datasets_mod.remove_dataset(rec["id"]) is True
    assert datasets_mod.get_dataset(rec["id"]) is None


def test_include_from_path_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_mod, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(datasets_mod, "DATASETS_REGISTRY_PATH", tmp_path / "hub_datasets.json")
    folder = tmp_path / "incoming"
    folder.mkdir()
    (folder / "a.txt").write_text("alpha example\n\nbeta example", encoding="utf-8")
    (folder / "b.jsonl").write_text('{"text":"gamma example"}\n', encoding="utf-8")
    rec = datasets_mod.include_from_path(name="Mixed folder", path=str(folder), split="blank_line")
    texts = datasets_mod.load_dataset_texts(rec["id"])
    assert rec["sample_count"] >= 3
    assert any("gamma" in t for t in texts)
