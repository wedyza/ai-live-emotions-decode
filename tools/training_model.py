from pathlib import Path
import torch
# --- Загрузка предобученной HSEmotion (EfficientNet, AffectNet) ---------------------------
# Порядок классов HSEmotion (idx_to_class из пакета hsemotion).
HSE_CLASSES_8 = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
HSE_CLASSES_7 = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
HSE_MODELS = {
    # имя весов           : (архитектура timm, размер входа, классы)
    "enet_b0_8_best_vgaf": ("tf_efficientnet_b0", 224, HSE_CLASSES_8),
    "enet_b0_8_best_afew": ("tf_efficientnet_b0", 224, HSE_CLASSES_8),
    "enet_b0_8_va_mtl":    ("tf_efficientnet_b0", 224, HSE_CLASSES_8),  # +2 выхода valence/arousal
    "enet_b2_8":           ("tf_efficientnet_b2", 260, HSE_CLASSES_8),
    "enet_b2_7":           ("tf_efficientnet_b2", 260, HSE_CLASSES_7),
}
HSE_URL = "https://github.com/av-savchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/{name}.pt"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_hsemotion(name: str, weights_dir: Path) -> tuple["torch.nn.Module", int, list[str]]:
    """Возвращает (модель timm с весами HSEmotion, размер входа, классы в порядке выходов).

    Автор выложил веса как целиком сохранённую модель (pickle), поэтому первая загрузка
    идёт с weights_only=False — это исполняет код из файла, качайте только из репозитория автора.
    Дальше веса кэшируются как обычный state_dict и грузятся безопасно (weights_only=True).
    """
    import timm
    arch, size, classes = HSE_MODELS[name]
    weights_dir.mkdir(parents=True, exist_ok=True)
    sd_path = weights_dir / f"{name}.state_dict.pth"
    if not sd_path.exists():
        pt_path = weights_dir / f"{name}.pt"
        if not pt_path.exists():
            print("downloading", HSE_URL.format(name=name))
            torch.hub.download_url_to_file(HSE_URL.format(name=name), str(pt_path))
        legacy = torch.load(pt_path, map_location="cpu", weights_only=False)
        sd = {k.replace("classifier.0.", "classifier."): v for k, v in legacy.state_dict().items()}
        torch.save(sd, sd_path)
    sd = torch.load(sd_path, map_location="cpu", weights_only=True)
    model = timm.create_model(arch, pretrained=False, num_classes=sd["classifier.weight"].shape[0])
    model.load_state_dict(sd, strict=True)
    return model.eval(), size, classes


def restrict_head(model: "torch.nn.Module", src_classes: list[str], dst_classes: list[str]) -> "torch.nn.Module":
    """Оставляет в последнем слое только строки нужных классов (в порядке dst_classes).

    Так дообучение стартует ровно с предобученной модели, а у VA-моделей отбрасываются
    выходы valence/arousal.
    """
    missing = [c for c in dst_classes if c not in src_classes]
    assert not missing, f"классов {missing} нет в предобученной модели: {src_classes}"
    idx = torch.tensor([src_classes.index(c) for c in dst_classes])
    old = model.classifier
    new = torch.nn.Linear(old.in_features, len(dst_classes))
    with torch.no_grad():
        new.weight.copy_(old.weight[idx])
        new.bias.copy_(old.bias[idx])
    model.classifier = new
    return model
