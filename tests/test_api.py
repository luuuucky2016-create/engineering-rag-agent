import pytest
from fastapi.testclient import TestClient

from app.api import app, get_assistant


class FakeAssistant:
    def __init__(self):
        self.calls = []

    def ask(self, message, thread_id="default"):
        self.calls.append((message, thread_id))
        return {"answer": f"echo: {message}", "sources": ["voltage_drop.md#1"], "tools_used": ["search_knowledge_base"], "thread_id": thread_id}


@pytest.fixture
def client_and_fake():
    fake = FakeAssistant()
    app.dependency_overrides[get_assistant] = lambda: fake
    yield TestClient(app), fake
    app.dependency_overrides.clear()


def test_health(client_and_fake):
    client, _ = client_and_fake
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_chat_returns_answer_sources_and_thread(client_and_fake):
    client, fake = client_and_fake
    r = client.post("/chat", json={"message": "voltage drop?", "thread_id": "abc"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "echo: voltage drop?"
    assert body["sources"] == ["voltage_drop.md#1"] and body["thread_id"] == "abc"


def test_chat_generates_thread_id_and_validates_input(client_and_fake):
    client, _ = client_and_fake
    assert client.post("/chat", json={"message": "hi"}).json()["thread_id"]
    assert client.post("/chat", json={"message": ""}).status_code == 422
