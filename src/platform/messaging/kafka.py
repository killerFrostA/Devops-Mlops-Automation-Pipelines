from src.contracts.events import EventEnvelope
from src.platform.errors import IntegrationNotConfiguredError


class KafkaPublisher:
    async def publish(self, event: EventEnvelope) -> None:
        raise IntegrationNotConfiguredError(
            "Implement validated Kafka publication, topic ACLs and delivery acknowledgements"
        )
