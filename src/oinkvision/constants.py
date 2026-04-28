LABELS = ["bad_posture", "bumps", "soft_pastern", "x_shape"]
CAMERAS = ["top", "right", "left", "rear"]


def get_active_labels(config: dict | None = None) -> list[str]:
    if not config:
        return list(LABELS)
    model_cfg = dict(config.get("model", {}))
    labels = model_cfg.get("active_labels")
    if not labels:
        return list(LABELS)
    return [str(label) for label in labels]
