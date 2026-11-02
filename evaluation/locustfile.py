"""Optional interactive/sustained load test against a running backend.

Usage::

    .venv\\Scripts\\python -m locust -f evaluation/locustfile.py --host http://127.0.0.1:8000

Then open the Locust web UI (default http://localhost:8089) to choose user
count and spawn rate. This is a supplementary tool for ad-hoc/sustained load
testing; ``evaluation/performance_eval.py`` is the scripted, reproducible
measurement used for the evaluation report.
"""

from __future__ import annotations

from locust import HttpUser, between, task

CHAT_MESSAGES = (
    "What parking slots are available?",
    "How much does a standard space cost?",
    "What are your opening hours?",
    "What are your parking rules?",
    "Where is the entrance?",
    "Can I cancel my booking for free?",
)


class ChatbotUser(HttpUser):
    """Simulates a user sending chat messages, reusing one session per user."""

    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        self.session_id: str | None = None

    @task(3)
    def send_chat_message(self) -> None:
        index = self.environment.runner.stats.num_requests % len(CHAT_MESSAGES)
        payload: dict[str, object] = {"message": CHAT_MESSAGES[index]}
        if self.session_id:
            payload["session_id"] = self.session_id
        with self.client.post("/chat", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                self.session_id = response.json().get("session_id")
            else:
                response.failure(f"unexpected status {response.status_code}")

    @task(1)
    def check_availability(self) -> None:
        self.client.get("/availability")

    @task(1)
    def health_check(self) -> None:
        self.client.get("/health")
