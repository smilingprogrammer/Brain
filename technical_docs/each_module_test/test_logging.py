# test_logging.py
from core.logging import setup_logging, get_logger, cognitive_logger

# Setup logging
setup_logging("DEBUG")

# Get a logger
logger = get_logger("test_module")

# Test basic logging
logger.info("test_event", module="test", action="started", value=42)
logger.warning("test_warning", issue="low_memory", threshold=0.8)
logger.error("test_error", error="connection_failed", retry_count=3)

# Test cognitive event logger
cognitive_logger.log_brain_event(
    event_type="reasoning_complete",
    region="prefrontal_cortex",
    data={"task": "solve_problem", "success": True},
    latency_ms=125.5
)

cognitive_logger.log_reasoning_path(
    problem="If A then B, A is true",
    paths_explored=["deductive", "logical"],
    final_conclusion="B is true",
    confidence=0.95
)

# Export session log
cognitive_logger.export_session_log("test_session.json")
print("Logs exported to test_session.json")



# Expected Output
# {"event": "test_event", "module": "test", "action": "started", "value": 42, "timestamp": "2024-01-01T12:00:00"}
# {"event": "test_warning", "issue": "low_memory", "threshold": 0.8, "timestamp": "2024-01-01T12:00:01"}
# {"event": "brain_event", "brain_region": "prefrontal_cortex", "latency_ms": 125.5}
# Logs exported to test_session.json