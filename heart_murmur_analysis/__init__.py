def load_model(*args, **kwargs):
    from .classification import load_model as _load_model
    return _load_model(*args, **kwargs)

def HeartSoundClassifier(*args, **kwargs):
    from .classification import HeartSoundClassifier as _HeartSoundClassifier
    return _HeartSoundClassifier(*args, **kwargs)

def HeartbeatAnalyzer(*args, **kwargs):
    from .signal_processing import HeartbeatAnalyzer as _HeartbeatAnalyzer
    return _HeartbeatAnalyzer(*args, **kwargs)

def build_heartbeat_agent(*args, **kwargs):
    from .agent.heartbeat_agent import build_heartbeat_agent as _build_heartbeat_agent
    return _build_heartbeat_agent(*args, **kwargs)

def export_json(*args, **kwargs):
    from .utils import export_json as _export_json
    return _export_json(*args, **kwargs)

def generate_hospital_report(*args, **kwargs):
    from .report_generator import generate_hospital_report as _generate_hospital_report
    return _generate_hospital_report(*args, **kwargs)

__version__ = "0.1.9"
