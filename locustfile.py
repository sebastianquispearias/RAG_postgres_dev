import random
import time

from locust import HttpUser, between, task


class ChatUser(HttpUser):
    wait_time = between(5, 20)

    @task
    def ask_question(self):
        self.client.get("/")
        time.sleep(5)
        self.client.post(
            "/chat",
            json={
                "messages": [
                    {
                        "content": random.choice(
                            ["Cómo evolucionó el consumo de diesel en enero de 2025?", "Hubo un pico en Custo Combustivel durante febrero?", "Resume el abastecimiento del vehículo 103001 en marzo?","¿Cuál fue el costo promedio de combustible el mes pasado?"]
                        ),
                        "role": "user",
                    }
                ],
                "context": {
                    "overrides": {"use_advanced_flow": True, "top": 3, "retrieval_mode": "hybrid", "temperature": 0.3}
                },
            },
        )
        time.sleep(5)
        self.client.post(
            "/chat",
            json={
                "messages": [
                    {"content": "Best shoe for hiking?", "role": "user"},
                    {
                        "content": "For the best shoe for hiking, I recommend the Trailblaze Steel-Blue Hiking Shoes.",
                        "role": "assistant",
                    },
                    {"content": "any other options?", "role": "user"},
                ],
                "context": {
                    "overrides": {"use_advanced_flow": True, "top": 3, "retrieval_mode": "hybrid", "temperature": 0.3}
                },
            },
        )
