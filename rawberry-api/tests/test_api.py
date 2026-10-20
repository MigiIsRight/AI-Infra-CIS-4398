import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_chat_service
from app.config import Settings
from app.main import create_app


def test_create_app_uses_isolated_storage():
    first_app = create_app()
    second_app = create_app()

    first_store = first_app.state.store
    second_store = second_app.state.store

    assert first_store is not second_store

    first_store.add({"id": "one", "text": "alpha", "metadata": {}})
    assert len(first_store.list_items()) == 1
    assert len(second_store.list_items()) == 0


def test_root_has_typed_contract():
    app = create_app()
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["message"] == "RAWBerry API is running"
    assert "/health" in payload["available_endpoints"]


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ingest_and_get_round_trip():
    app = create_app()
    client = TestClient(app)

    ingest_response = client.post(
        "/ingest",
        json={"userid": 1, "text": "hello world", "metadata": {"source": "demo"}},
    )

    assert ingest_response.status_code == 201
    payload = ingest_response.json()
    assert payload["message"] == "Item ingested successfully"
    assert payload["item"]["text"] == "hello world"
    assert payload["count"] == 1

    get_response = client.get("/get")
    assert get_response.status_code == 200
    assert get_response.json()["count"] == 1
    assert len(app.state.store.list_chunks()) == 1
    assert app.state.store.list_chunks()[0]["status"] == "indexed"


def test_chat_uses_typed_reply(monkeypatch):
    monkeypatch.setenv("USE_MOCK_CHAT", "true")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT_ID", raising=False)

    app = create_app()
    client = TestClient(app)
    client.post(
        "/ingest",
        json={"userid": 1, "text": "first item"},
    )

    response = client.post(
        "/chat",
        json={
            "userid": 1,
            "data": {"window": 3, "agent": "test"},
            "message": "hello",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["reply"] == "You said: hello"
    assert len(payload["recent_items"]) == 1


def test_chat_stream_emits_lifecycle_status_and_final_response(monkeypatch):
    monkeypatch.setenv("USE_MOCK_CHAT", "true")
    app = create_app()
    client = TestClient(app)

    with client.stream(
        "POST",
        "/chat/stream",
        json={
            "userid": 1,
            "data": {"window": 3, "agent": "test"},
            "message": "hello",
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert body.index('"status":"received"') < body.index('"status":"generating"')
    assert body.index('"status":"generating"') < body.index('"status":"completed"')
    assert '"reply":"You said: hello"' in body


def test_chat_stream_emits_failure_status_without_exposing_exception():
    from app.api.routes import get_chat_service

    class FailingChatService:
        def generate_reply(self, request):
            raise RuntimeError("private provider detail")

    app = create_app()
    app.dependency_overrides[get_chat_service] = lambda: FailingChatService()
    client = TestClient(app)

    response = client.post(
        "/chat/stream",
        json={
            "userid": 1,
            "data": {"window": 3, "agent": "test"},
            "message": "hello",
        },
    )

    assert response.status_code == 200
    assert '"status":"failed"' in response.text
    assert "The response could not be generated." in response.text
    assert "private provider detail" not in response.text


def test_system_prompt_can_be_read_updated_and_cleared():
    client = TestClient(create_app())

    initial = client.get("/users/42/system-prompt")
    assert initial.status_code == 200
    assert initial.json() == {"userid": 42, "prompt": ""}

    updated = client.put("/users/42/system-prompt", json={"prompt": "Be concise."})
    assert updated.status_code == 200
    assert updated.json() == {"userid": 42, "prompt": "Be concise."}
    assert client.get("/users/43/system-prompt").json()["prompt"] == ""

    cleared = client.put("/users/42/system-prompt", json={"prompt": ""})
    assert cleared.status_code == 200
    assert cleared.json()["prompt"] == ""


def test_system_prompt_rejects_values_over_limit():
    client = TestClient(create_app())

    response = client.put("/users/42/system-prompt", json={"prompt": "x" * 2001})

    assert response.status_code == 422


def test_chat_includes_saved_user_system_prompt():
    client = TestClient(create_app())
    client.put("/users/42/system-prompt", json={"prompt": "Be concise."})

    response = client.post(
        "/chat",
        json={
            "userid": 42,
            "data": {"window": 3, "agent": "test"},
            "message": "Explain this.",
        },
    )

    assert response.status_code == 200
    assert "Be concise." in response.json()["reply"]
    assert "Explain this." in response.json()["reply"]


def test_invalid_message_is_rejected():
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/chat",
        json={
            "userid": 1,
            "data": {"window": 3, "agent": "test"},
            "message": "",
        },
    )
    assert response.status_code == 422


def test_chat_route_uses_mock_fallback_when_disabled(monkeypatch):
    monkeypatch.setenv("USE_MOCK_CHAT", "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "userid": 1,
            "data": {"window": 3, "agent": "test"},
            "message": "fallback check",
        },
    )
    assert response.status_code == 200
    assert response.json()["reply"] == "You said: fallback check"


def test_chat_route_uses_configured_gemini_client(monkeypatch):
    monkeypatch.setenv("USE_MOCK_CHAT", "false")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")

    from app.services import chat_service as chat_service_module

    fake_llm = MagicMock()
    fake_llm.generate_reply.return_value = "gemini reply"
    monkeypatch.setattr(chat_service_module, "GeminiChatClient", MagicMock(return_value=fake_llm))

    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "userid": 1,
            "data": {"window": 3, "agent": "test"},
            "message": "hello",
        },
    )
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert [e['status'] for e in events] == ['received', 'generating', 'completed']
    assert len({e['request_id'] for e in events}) == 1
    assert events[-1]['response']['reply'] == 'You said: hello'
    assert set(events[-1]['response']) == {'conversation_id', 'turn_id', 'reply'}
    assert len(client.get('/conversations/' + identifier + '/messages').json()) == 2


@pytest.mark.parametrize('stream', [False, True])
def test_generation_failure_is_recorded_and_can_be_retried(stream):
    store = InMemoryStore()
    app = create_app(store=store)
    llm = MagicMock()
    llm.generate_reply.side_effect = RuntimeError('private provider details')
    app.dependency_overrides[get_chat_service] = lambda: ChatService(store, llm_client=llm)
    with TestClient(app, headers={'X-User-ID': 'alice'}) as client:
        identifier = conversation(client)
        response = ask(client, identifier, stream=stream)
        assert response.status_code == (200 if stream else 502)
        assert 'private provider details' not in response.text
        if stream:
            assert '"status":"failed"' in response.text
            assert '"status":"completed"' not in response.text
        messages = client.get('/conversations/' + identifier + '/messages').json()
        assert len(messages) == 1 and messages[0]['status'] == 'failed'
        llm.generate_reply.side_effect = None
        llm.generate_reply.return_value = 'recovered'
        assert ask(client, identifier).status_code == 200
        messages = client.get('/conversations/' + identifier + '/messages').json()
        assert [m['status'] for m in messages] == ['failed', 'completed', 'completed']


def test_concurrent_turns_and_abandoned_turn_recovery(client):
    store = client.app.state.store
    identifier = conversation(client)
    old = store.start_turn('alice', identifier, 'first')
    assert ask(client, identifier).status_code == 409
    assert ask(client, identifier, stream=True).status_code == 409
    store._turns[identifier][0]['created_at'] = datetime.now(timezone.utc) - timedelta(minutes=6)
    assert ask(client, identifier).status_code == 200
    with pytest.raises(ConversationBusy):
        store.finish_turn('alice', identifier, old['id'], 'late response')
    assert store.list_messages('alice', identifier)[0]['status'] == 'failed'


def test_preferences_update_clear_and_validation(client):
    assert client.get('/me/system-prompt').json()['prompt'] == ''
    assert client.put('/me/system-prompt', json={'prompt': 'Concise'}).status_code == 200
    assert client.get('/me/system-prompt', headers={'X-User-ID': 'bob'}).json()['prompt'] == ''
    assert client.put('/me/system-prompt', json={'prompt': ''}).status_code == 200
    assert client.get('/me/system-prompt').json()['prompt'] == ''
    assert client.put('/me/system-prompt', json={'prompt': 'x' * 2001}).status_code == 422
    assert client.post('/chat', json={'conversation_id': conversation(client), 'message': ''}).status_code == 422


def test_upload_retains_bytes_and_txt_chunks(client):
    response = upload(client, b'hello gamma')
    assert response.status_code == 200
    document = response.json()[0]
    assert document['chunk_count'] == 1 and document['embedding_status'] == 'pending'
    store = client.app.state.store
    assert store.get_document_content('alice', document['document_id']) == b'hello gamma'
    assert store.list_chunks('alice')[0]['page_number'] is None
    assert store.list_chunks('bob') == []
    assert upload(client, b'%PDF-1.7 sample', name='paper.pdf').json()[0]['chunk_count'] == 0


@pytest.mark.parametrize('name,content,status,code', [
    ('a.csv', b'a,b', 400, 'INVALID_FILE_TYPE'),
    ('a.txt', b'', 400, 'EMPTY_FILE'),
    ('a.txt', b'\xff\xfe', 400, 'INVALID_TEXT_ENCODING'),
    ('a.pdf', b'not pdf', 400, 'INVALID_FILE_CONTENT'),
    ('a.docx', b'not zip', 400, 'INVALID_FILE_CONTENT'),
])
def test_invalid_uploads(client, name, content, status, code):
    response = upload(client, content, name=name)
    assert response.status_code == status
    assert response.json()['error_code'] == code
    assert client.app.state.store.list_documents('alice') == []


def test_upload_batch_validation_and_size_limit(client, monkeypatch):
    response = client.post('/upload', files=[('files', ('a.txt', b'hello', 'text/plain')),
                                            ('files', ('b.pdf', b'invalid', 'application/pdf'))])
    assert response.status_code == 400
    assert client.app.state.store.list_documents('alice') == []
    from app.services.ingest_service import IngestService

    monkeypatch.setattr(IngestService, "MAX_FILE_SIZE_BYTES", 4)
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/upload",
        files=[("files", ("large.txt", b"12345", "text/plain"))],
    )

    assert response.status_code == 413
    assert response.json()["error_code"] == "FILE_TOO_LARGE"
    assert app.state.store.list_documents() == []


def test_upload_endpoint_rejects_non_utf8_text():
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/upload",
        files=[("files", ("invalid.txt", b"\xff\xfe", "text/plain"))],
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_TEXT_ENCODING"


def test_chat_uses_retrieved_document_chunks_as_context():
    app = create_app()
    client = TestClient(app)

    upload = client.post(
        "/upload",
        files=[("files", ("sample.txt", b"alpha beta gamma delta", "text/plain"))],
        data={"owner_id": "user-123"},
    )
    assert upload.status_code == 200

    response = client.post(
        "/chat",
        json={
            "userid": 1,
            "data": {"window": 3, "agent": "test"},
            "message": "What is gamma?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "gamma" in payload["reply"].lower()
    assert "alpha beta gamma delta" in payload["reply"].lower()


def test_chat_combines_retrieved_context_and_saved_response_preferences():
    app = create_app()
    client = TestClient(app)
    client.put("/users/7/system-prompt", json={"prompt": "Answer in one sentence."})
    client.post(
        "/ingest",
        json={"userid": 7, "text": "The moon orbits Earth every 27 days."},
    )

    response = client.post(
        "/chat",
        json={
            "userid": 7,
            "data": {"window": 3, "agent": "test"},
            "message": "How long does the moon orbit Earth?",
        },
    )

    assert response.status_code == 200
    reply = response.json()["reply"]
    assert "Answer in one sentence." in reply
    assert "The moon orbits Earth every 27 days." in reply
    assert "How long does the moon orbit Earth?" in reply
