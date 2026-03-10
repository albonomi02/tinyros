"""Network configuration helpers for tinyros multiprocess benchmark."""

from tinyros import TinyNetworkConfig, TinyNodeDescription, TinySubscription


class Nodes:
    """Node names used by the benchmark."""

    PUBLISHER = "publisher"
    SUBSCRIBER = "subscriber"


class Topics:
    """Topic names used by the benchmark."""

    PAYLOAD = "payload"
    READY = "ready"


def build_network_config(*, pub_port: int, sub_port: int) -> TinyNetworkConfig:
    """Build a 2-node network (publisher -> subscriber)."""
    return TinyNetworkConfig(
        nodes={
            Nodes.PUBLISHER: TinyNodeDescription(port=pub_port, host="localhost"),
            Nodes.SUBSCRIBER: TinyNodeDescription(port=sub_port, host="localhost"),
        },
        connections={
            Nodes.PUBLISHER: {
                Topics.PAYLOAD: [
                    TinySubscription(actor=Nodes.SUBSCRIBER, cb_name="on_msg")
                ]
            },
            Nodes.SUBSCRIBER: {
                Topics.READY: [
                    TinySubscription(actor=Nodes.PUBLISHER, cb_name="on_ready")
                ]
            }
        },
    )
