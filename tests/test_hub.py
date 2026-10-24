"""Hub registry, chat prompt formatting, and agent JSON parsing."""

from src.agent.backends import parse_agent_json
from src.hub.registry import get_model, remove_model, upsert_model
from src.hub.runtime import format_chat_prompt


def test_format_chat_prompt_includes_history_and_system():
    prompt = format_chat_prompt(
        "hello",
        history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}],
        system_prompt="Be brief",
    )
    assert "System: Be brief" in prompt
    assert "User: hi" in prompt
    assert "Assistant: hey" in prompt
    assert prompt.strip().endswith("Assistant:")


def test_parse_agent_json_extracts_tool_call():
    decision = parse_agent_json('noise {"thought": "look", "tool": "list_directory", "params": {"subpath": ""}, "is_finish": false} trailing')
    assert decision.tool_name == "list_directory"
    assert decision.tool_params["subpath"] == ""
    assert decision.is_finish is False


def test_registry_upsert_and_get(tmp_path, monkeypatch):
    from src.hub import paths, registry

    monkeypatch.setattr(paths, "REGISTRY_PATH", tmp_path / "hub_registry.json")
    monkeypatch.setattr(paths, "CHECKPOINTS_DIR", tmp_path / "ckpts")
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "hub_registry.json")
    monkeypatch.setattr(registry, "CHECKPOINTS_DIR", tmp_path / "ckpts")
    (tmp_path / "ckpts").mkdir()

    rec = upsert_model(
        {
            "id": "demo-ollama",
            "name": "demo",
            "provider": "ollama",
            "kind": "ollama",
            "origin": "llama3",
            "status": "registered",
        }
    )
    assert rec["id"] == "demo-ollama"
    assert get_model("demo-ollama")["origin"] == "llama3"
    assert remove_model("demo-ollama") is True
    assert get_model("demo-ollama") is None
