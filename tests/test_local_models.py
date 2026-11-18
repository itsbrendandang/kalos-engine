"""Local transport, diagnostics, and reply cleanup regressions."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from kalos.assist.llm import OpenAICompatChat, get_chat_model, model_info
from kalos.normalize.config import load_config
from kalos.normalize.llm import _extract_json_object
from kalos.providers import ollama_provider as ollama
from kalos.providers import openai_compatible_provider as openai
from kalos.providers.check import main


def test_openai_does_not_follow_redirects(monkeypatch):
    paths = []

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/redirected")
            self.end_headers()

        do_POST = do_GET

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Redirect) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        monkeypatch.setenv("KALOS_OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
        monkeypatch.setenv("KALOS_OPENAI_MODEL", "qwen")
        monkeypatch.setenv("KALOS_OPENAI_API_KEY", "test-key")
        try:
            with pytest.raises(openai.OpenAICompatUnavailable):
                openai.OpenAICompatibleProvider().chat("system", "prompt")
            assert openai.OpenAICompatibleProvider().probe() == {"reachable": False, "model_present": False}
            assert paths == ["/v1/chat/completions", "/v1/models"]
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("module,cls,setting,listing", [
    (ollama, ollama.OllamaProvider, "KALOS_OLLAMA_MODEL", {"models": [{"name": "qwen"}]}),
    (openai, openai.OpenAICompatibleProvider, "KALOS_OPENAI_MODEL", {"data": [{"id": "qwen"}]}),
])
def test_probe(monkeypatch, module, cls, setting, listing):
    monkeypatch.setenv(setting, "qwen")
    calls = []

    def respond(request, timeout):
        calls.append((request.full_url, timeout))
        return json.dumps(listing).encode()

    monkeypatch.setattr(module, "_open", respond)
    assert cls().probe() == {"reachable": True, "model_present": True}
    assert calls[0][1] == 2.0
    monkeypatch.setenv(setting, "missing")
    assert cls().probe() == {"reachable": True, "model_present": False}
    monkeypatch.setattr(module, "_open", lambda *a: b"{}")
    assert cls().probe() == {"reachable": True, "model_present": False}
    monkeypatch.setattr(module, "_open", lambda *a: b"invalid")
    assert cls().probe() == {"reachable": False, "model_present": False}

    def fail(*args):
        raise OSError("offline")

    monkeypatch.setattr(module, "_open", fail)
    assert cls().probe() == {"reachable": False, "model_present": False}


def test_openai_assistant_and_normalize_startup(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("KALOS_OPENAI_MODEL", "qwen")
    model = get_chat_model()
    assert isinstance(model, OpenAICompatChat)
    assert model_info() == {"provider": "openai", "model": "qwen", "local": True}
    assert load_config().enabled_live is False
    from kalos.portal.config import check_env_settings

    check_env_settings()


@pytest.mark.parametrize("flag", ["", "0", "false", "1", "true", "YES", "on"])
def test_openai_remote_opt_in(monkeypatch, flag):
    monkeypatch.setenv("KALOS_OPENAI_BASE_URL", "http://192.0.2.1:8080/v1")
    monkeypatch.setenv("KALOS_OPENAI_MODEL", "qwen")
    monkeypatch.setenv("KALOS_OPENAI_ALLOW_REMOTE", flag)
    calls = []

    def respond(request, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith("/models"):
            return b'{"data":[{"id":"qwen"}]}'
        return b'{"choices":[{"message":{"content":"answer"}}]}'

    monkeypatch.setattr(openai, "_open", respond)
    provider = openai.OpenAICompatibleProvider()
    if flag.lower() in {"1", "true", "yes", "on"}:
        assert provider.chat("system", "prompt") == "answer"
        assert provider.probe() == {"reachable": True, "model_present": True}
        assert len(calls) == 2
    else:
        with pytest.raises(openai.OpenAICompatUnavailable, match="KALOS_OPENAI_ALLOW_REMOTE"):
            provider.chat("system", "prompt")
        assert provider.probe() == {"reachable": False, "model_present": False}
        assert calls == []


def test_chat_cleanup_and_openai_request(monkeypatch):
    monkeypatch.setenv("KALOS_OPENAI_MODEL", "qwen")
    monkeypatch.setenv("KALOS_OPENAI_API_KEY", "test-key")

    def respond(request, timeout):
        assert request.get_header("Authorization") == "Bearer test-key"
        assert json.loads(request.data)["model"] == "qwen"
        return b'{"choices":[{"message":{"content":"<think>secret</think>answer","reasoning_content":"hidden"}}]}'

    monkeypatch.setattr(openai, "_open", respond)
    assert openai.OpenAICompatibleProvider().chat("system", "prompt") == "answer"
    monkeypatch.setattr(ollama, "_open", lambda *a: b'{"message":{"content":"<think>secret</think>answer"}}')
    assert ollama.OllamaProvider().chat("system", "prompt") == "answer"


@pytest.mark.parametrize("reply", [b"{}", b"invalid", b'{"choices":[]}', b'{"choices":[{"message":{"content":""}}]}'])
def test_openai_bad_reply(monkeypatch, reply):
    monkeypatch.setenv("KALOS_OPENAI_MODEL", "qwen")
    monkeypatch.setattr(openai, "_open", lambda *a: reply)
    with pytest.raises(openai.OpenAICompatUnavailable):
        openai.OpenAICompatibleProvider().chat("system", "prompt")


@pytest.mark.parametrize("text,expected", [
    ('<think>{"wrong":true}</think>\n{"columns":[]}', {"columns": []}),
    ("<think>unfinished", None), ("", None), ("not JSON", None),
])
def test_normalize_thinking(text, expected):
    assert _extract_json_object(text) == expected


def test_shared_default(monkeypatch):
    monkeypatch.delenv("KALOS_OLLAMA_MODEL", raising=False)
    assert load_config().ollama_model == ollama.DEFAULT_MODEL == "qwen2.5:7b-instruct"


@pytest.mark.parametrize("selected,listed,present", [
    ("llama3.1", "llama3.1:latest", True),
    ("llama3.1:latest", "llama3.1", True),
    ("llama3.1", "llama3.1:8b", False),
    ("host:5000/library/llama3.1", "host:5000/library/llama3.1:latest", True),
])
def test_ollama_probe_default_tag(monkeypatch, selected, listed, present):
    monkeypatch.setenv("KALOS_OLLAMA_MODEL", selected)
    monkeypatch.setattr(ollama, "_open", lambda *a: json.dumps({"models": [{"name": listed}]}).encode())
    assert ollama.OllamaProvider().probe() == {"reachable": True, "model_present": present}


def test_cli_and_status_probe_only_on_request(monkeypatch, capsys):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("KALOS_OLLAMA_MODEL", "qwen")
    calls = []

    def probe(self):
        calls.append(True)
        return {"reachable": True, "model_present": False}

    monkeypatch.setattr(ollama.OllamaProvider, "probe", probe)
    model_info()
    assert calls == []
    assert model_info(probe=True)["reachable"] is True
    main()
    output = capsys.readouterr().out
    assert "Provider: ollama" in output
    assert "Model pulled: False" in output
    assert "ollama pull qwen" in output
