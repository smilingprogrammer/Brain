import structlog
import logging
import sys
from typing import Optional
import json
from datetime import datetime


def setup_logging(log_level: str = "INFO"):
    """Configure structured logging for the cognitive system"""

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)


class CognitiveEventLogger:
    """Specialized logger for cognitive events"""

    def __init__(self):
        self.logger = get_logger("cognitive_events")
        self.event_history = []

    def log_brain_event(self,
                        event_type: str,
                        region: str,
                        data: dict,
                        latency_ms: Optional[float] = None):
        """Log a brain region event"""

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "brain_region": region,
            "data": data
        }

        if latency_ms is not None:
            event["latency_ms"] = latency_ms

        self.event_history.append(event)
        self.logger.info("brain_event", **event)

    def log_reasoning_path(self,
                           problem: str,
                           paths_explored: list,
                           final_conclusion: str,
                           confidence: float):
        """Log a complete reasoning process"""

        self.logger.info(
            "reasoning_complete",
            problem=problem,
            paths_explored=paths_explored,
            conclusion=final_conclusion,
            confidence=confidence,
            path_count=len(paths_explored)
        )

    def export_session_log(self, filepath: str):
        """Export session events to file"""

        with open(filepath, 'w') as f:
            json.dump(self.event_history, f, indent=2)

        self.logger.info("session_exported", filepath=filepath, event_count=len(self.event_history))


# Global cognitive event logger
cognitive_logger = CognitiveEventLogger()