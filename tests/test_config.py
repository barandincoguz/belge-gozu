from belge_gozu.config import Settings


def test_defaults():
    s = Settings()
    assert s.retriever_model == "vidore/colSmol-500M"
    assert s.stage1_candidates == 200
    assert s.top_k == 5
    assert s.request_delay_s == 1.0


def test_env_override(monkeypatch):
    monkeypatch.setenv("BG_TOP_K", "3")
    assert Settings().top_k == 3


def test_env_file_and_google_alias(monkeypatch, tmp_path):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("BG_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GOOGLE_API_KEY=test-key-123\n")
    assert Settings().gemini_api_key == "test-key-123"
