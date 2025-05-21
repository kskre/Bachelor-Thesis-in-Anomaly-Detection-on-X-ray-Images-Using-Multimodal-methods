import os
import uuid
import re
import csv
from io import StringIO
import io
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForTokenClassification
import torch.nn as nn
import matplotlib.pyplot as plt
from textblob import TextBlob

"""
Inference pipeline for medical image and text symptom analysis.

Functions and classes:
  normalize_phrase(phrase): correct typos using TextBlob.
  has_symptom_entity(phrase): detect biomedical entities via NER pipeline.
  is_symptom_zero_shot(phrase, threshold, margin): classify text as symptom via zero-shot NLI.
  clean_symptom_phrase(phrase): split and clean individual symptom phrases.
  parse_symptom_list(symptom_list_text): parse CSV-like symptom lists.
  preprocess_text(input_text): normalize or mark no-symptoms token.
  filter_symptom_phrases_with_policy(text, tokenizer, policy_model, threshold, device): 
  multi-stage symptom filtering (parsing, NER, zero-shot, CNN policy).

  SymptomPolicyCNN: CNN-based policy network for symptom filtering.
  ImageEncoder: CNN encoder (DenseNet121) for image features.
  TextEncoder: BERT-based encoder for text symptom embeddings.
  JointClassifier: fully connected network fusing image + text embeddings.

  load_models_symptom(): load pretrained symptom models (policy, encoders, classifier, thresholds).
  predict(image_input, symptoms_text, label_names, box_size, model_dict): run end-to-end inference and Grad-CAM to produce bounding boxes.
  load_images(image_paths): load and return PIL images.

Constants:
  CONFIG: model/configuration settings.
  SYMPTOM_LABELS, NO_SYMPTOMS_TOKEN, LABEL_NAMES: label definitions.
  tokenizer, model, ner_pipeline, zero_shot_clf: pretrained pipelines.
"""

CONFIG = {
    "embed_dim": 256,
    "max_length": 64,
    "policy_threshold": 0.3,
    "model_path": "best_model_symptoms.pt",
    "model_path_finetuned": "fine_tuned_model_text_symptoms.pt",
    "thresholds_path": "optimized_thresholds.npy",
    "image_save_dir": "results",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}
os.makedirs(CONFIG["image_save_dir"], exist_ok=True)

SYMPTOM_LABELS = {
    "SIGN_SYMPTOM",
    "SEVERITY",
    "DISEASE_DISORDER",
}
NO_SYMPTOMS_TOKEN = "[NO_SYMPTOMS_PROVIDED]"

LABEL_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Effusion',
    'Emphysema', 'Fibrosis', 'Infiltration', 'Pneumonia', 'Pneumothorax'
]

IMAGE_PATHS = [
    'raw_files/test.png',
    'raw_files/test_image1.png',
]


tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
model     = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

def normalize_phrase(phrase):

    corrected = str(TextBlob(phrase).correct())
    return corrected.strip()

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=0   
)

def has_symptom_entity(phrase):
    print(phrase)
    ents = ner_pipeline(phrase)
    print(ents)
    return any(ent["entity_group"].upper() in SYMPTOM_LABELS for ent in ents)

zero_shot_clf = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1,
    multi_label=False
)

def is_symptom_zero_shot(phrase, threshold= 0.8, margin = 0.3):
    res = zero_shot_clf(
        phrase,
        candidate_labels=[
            "This text describes a medical symptom.",
            "This text does not describe a medical symptom."
        ]
    )
    labels = res["labels"]
    scores = res["scores"]
    idx_sym = labels.index("This text describes a medical symptom.")
    idx_not = labels.index("This text does not describe a medical symptom.")
    sym_score = scores[idx_sym]
    not_score = scores[idx_not]
    return (sym_score >= threshold) and ((sym_score - not_score) >= margin)

def clean_symptom_phrase(phrase):
    phrase = phrase.strip(" '\"").strip()
    phrase = re.sub(r"\s+", " ", phrase)
    phrase = re.sub(r"[“”]", '"', phrase)
    phrase = re.sub(r"[‘’]", "'", phrase)
    phrase = re.sub(r"[—–−]", "-", phrase)

    paren_match = re.match(r"^(.*)\((.*?)\)(.*)$", phrase)
    if paren_match:
        before_and_after = (paren_match.group(1) + paren_match.group(3)).strip(" ,.;:")
        inside_parentheses = paren_match.group(2).strip(" ,.;:")
        return [p for p in (before_and_after, inside_parentheses) if p]
    return [phrase.strip(" ,.;:")]


def parse_symptom_list(symptom_list_text):
    normalized_text = symptom_list_text.strip().replace("'", '"')
    csv_reader = csv.reader(StringIO(normalized_text), skipinitialspace=True)
    
    try:
        raw_fields = next(csv_reader)
    except StopIteration:
        return []
    
    cleaned_symptoms = []
    for raw_field in raw_fields:
        cleaned_symptoms += clean_symptom_phrase(raw_field)
    
    return [symptom for symptom in cleaned_symptoms if symptom]


def preprocess_text(input_text):
    stripped_text = input_text.strip()
    normalized_key = stripped_text.upper().replace(" ", "")
    if not stripped_text or normalized_key == NO_SYMPTOMS_TOKEN:
        return NO_SYMPTOMS_TOKEN
    symptoms = parse_symptom_list(stripped_text)
    joined = ", ".join(symptoms)
    return joined or NO_SYMPTOMS_TOKEN

def is_no_symptom(input_text):
    return input_text.strip().upper().replace(" ", "") == NO_SYMPTOMS_TOKEN


class SymptomPolicyCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs1 = nn.ModuleList([
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        self.fc = nn.Linear(hidden_dim * len(self.convs1), 2)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  # [B, T, D]

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        x = x.permute(0, 2, 1)  # [B, D, T]
        conv_outs = [F.relu(conv(x)) for conv in self.convs1]
        pooled = [F.adaptive_max_pool1d(c, 1).squeeze(-1) for c in conv_outs]
        concat = torch.cat(pooled, dim=1)
        logits = self.fc(concat)
        return logits, concat

def filter_symptom_phrases_with_policy(text, tokenizer, policy_model, threshold, device):
    policy_model.eval()
    #print(f"\n>>> RAW input: {repr(text)}")
    if is_no_symptom(text):
    #    print(">>> Detected as NO_SYMPTOMS_PROVIDED (early exit).")
        return text, [], []
    
    phrases = parse_symptom_list(text)
    #print(f">>> After parsing: {phrases}")

    phrases = [normalize_phrase(p) for p in phrases]
    #print(f">>> After spellcheck/normalization: {phrases}")

    ner_candidates = []
    for p in phrases:
        has_ent = has_symptom_entity(p)
        print(f"    - NER for '{p}': {has_ent}")
        if has_ent:
            ner_candidates.append(p)
   # print(f">>> After NER filter: {ner_candidates}")

    sem_filtered = []
    for p in ner_candidates:
        is_sym = is_symptom_zero_shot(p)
        print(f"    - Zero-shot for '{p}': {is_sym}")
        if is_sym:
            sem_filtered.append(p)
    #print(f">>> After zero-shot filter: {sem_filtered}")

    if not sem_filtered:
        print(">>> No symptoms left after semantic filtering.")
        return NO_SYMPTOMS_TOKEN, [], []
    
    # 5) policy CNN fine-filter
    enc = tokenizer(sem_filtered, padding="max_length", truncation=True,
                    max_length=32, return_tensors="pt").to(device)
    with torch.no_grad():
        logits, _ = policy_model(enc["input_ids"], enc.get("attention_mask"))
        probs = F.softmax(logits, dim=1)[:, 1]
    acts = (probs > threshold).tolist()
    #print(f">>> Policy logits: {logits}")
    #print(f">>> Policy probs: {probs.tolist()}")
    #print(f">>> Policy acts (above threshold): {acts}")

    keep = [p for p, a in zip(sem_filtered, acts) if a]
    out = ", ".join(keep) if keep else NO_SYMPTOMS_TOKEN
    print(f">>> FINAL filtered symptoms: {out}")

    return out, acts, [bool(a) for a in acts]

class ImageEncoder(nn.Module):
    def __init__(self, output_dim, normalize=True):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base.classifier.in_features, output_dim)
        self.normalize = normalize
    def forward(self, images, return_features=False):
        features = self.features(images)  # [B, C, H, W]
        pooled = self.pool(features).view(features.size(0), -1)
        embedding = self.fc(pooled)
        embedding = F.normalize(embedding, dim=-1) if self.normalize else embedding
    
        if return_features:
            return embedding, features
        return embedding

class TextEncoder(nn.Module):
    def __init__(self, output_dim, tokenizer, normalize=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.bert.resize_token_embeddings(len(tokenizer))
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.null_symptom_embedding = nn.Parameter(torch.randn(output_dim))
        self.normalize = normalize

    def forward(self, input_ids, attention_mask, null_symptom_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = output.last_hidden_state  # shape: [B, T, H]

        mask = attention_mask.unsqueeze(-1).float()
        masked_sum = torch.sum(token_embeddings * mask, dim=1)
        token_count = mask.sum(dim=1).clamp(min=1e-9)
        pooled = masked_sum / token_count  # mean pooling

        projected = self.fc(pooled)
        normalized = F.normalize(projected, dim=-1) if self.normalize else projected

        null_embeddings = self.null_symptom_embedding.unsqueeze(0).expand(normalized.size(0), -1)
        final = torch.where(null_symptom_mask.unsqueeze(-1), null_embeddings, normalized)
        return final


class JointClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image_features, text_features):
        fused_features = torch.cat([image_features, text_features], dim=-1)
        return self.classifier(fused_features)

image_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def load_models_symptom():
    device = CONFIG["device"]
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    policy = SymptomPolicyCNN(tokenizer.vocab_size, CONFIG["embed_dim"]).to(device)
    ck = torch.load(CONFIG["model_path"], map_location=device)
    policy.load_state_dict(ck["policy_model"])
    policy.eval()

    image_enc = ImageEncoder(CONFIG["embed_dim"]).to(device)
    text_enc = TextEncoder(CONFIG["embed_dim"], tokenizer).to(device)
    ck2 = torch.load(CONFIG["model_path_finetuned"], map_location=device)
    image_enc.load_state_dict(ck2["image_encoder"])
    text_enc.load_state_dict(ck2["text_encoder"])

    label_names = LABEL_NAMES

    clf = JointClassifier(CONFIG["embed_dim"]*2, len(label_names)).to(device)
    clf.load_state_dict(ck2["classifier"])
    image_enc.eval()
    text_enc.eval()
    clf.eval()

    thresholds = np.load(CONFIG["thresholds_path"])
    return {
        "tokenizer": tokenizer,
        "policy": policy,
        "image_enc": image_enc,
        "text_enc": text_enc,
        "clf": clf,
        "thresholds": thresholds,
        "label_names": label_names,
    }


def predict(image_input, symptoms_text, label_names, box_size=50, model_dict=None):
    device = CONFIG["device"]
    print(f">>> Incoming symptoms_text: {symptoms_text}")

    if model_dict is None:
        raise ValueError("Model dict must be passed!")

    tokenizer = model_dict["tokenizer"]
    policy = model_dict["policy"]
    image_enc = model_dict["image_enc"]
    text_enc = model_dict["text_enc"]
    clf = model_dict["clf"]
    thresholds = model_dict["thresholds"]
    label_names = model_dict["label_names"]

    filtered, _, _ = filter_symptom_phrases_with_policy(
        symptoms_text, tokenizer, policy,
        threshold=CONFIG["policy_threshold"], device=device
    )
    clean_text = preprocess_text(filtered)
    print(f">>> Filtered symptoms_text: {clean_text}")

    # Prepare image
    if isinstance(image_input, (bytes, bytearray)):
        raw = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, str):
        raw = Image.open(image_input).convert("RGB")
    else:
        raw = image_input.convert("RGB")
    inp = image_transform(raw).unsqueeze(0).to(device)

    # Attach hooks to final conv layer
    conv_feats, grads = None, None
    def forward_hook(module, inp_, out):
        nonlocal conv_feats
        conv_feats = out
        conv_feats.retain_grad()

    def backward_hook(module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0]

    target_layer = image_enc.features[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    feat_i, _ = image_enc(inp, return_features=True)
    enc = tokenizer(
        clean_text,
        padding="max_length", truncation=True,
        max_length=CONFIG["max_length"], return_tensors="pt"
    ).to(device)
    null_mask = torch.tensor([clean_text == "[NO_SYMPTOMS_PROVIDED]"], dtype=torch.bool, device=device)
    feat_t = text_enc(enc["input_ids"], enc["attention_mask"], null_mask)
    logits = clf(feat_i, feat_t).squeeze(0)

    # Multilabel prediction
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs > thresholds).astype(int)
    predicted = preds.copy()
    if not predicted.any():
        top = int(np.argmax(probs))
        predicted = np.zeros_like(predicted)
        predicted[top] = 1

    labels = [label_names[i] for i,p in enumerate(predicted) if p]
    probs_dict = {label_names[i]: float(probs[i]) for i,p in enumerate(predicted) if p}

    # Prepare colors for boxes
    cmap = plt.get_cmap('tab10')
    class_colors = {lbl: tuple((np.array(cmap(i))[:3]*255).astype(int).tolist())
                    for i,lbl in enumerate(labels)}

    # Prepare image for drawing
    raw_np = np.array(raw.resize((224,224)))
    boxed  = raw_np.copy()

    # Per-class Grad-CAM and draw a box around the max activation
    for cls in labels:
        cls_idx = label_names.index(cls)
        clf.zero_grad()
        logits[cls_idx].backward(retain_graph=True)

        fmap = conv_feats[0].cpu().detach().numpy()
        grad = grads[0].cpu().detach().numpy()
        weights = grad.mean(axis=(1,2))
        cam = np.zeros(fmap.shape[1:], dtype=np.float32)
        for i,w in enumerate(weights):
            cam += w * fmap[i]
        cam = np.clip(cam, 0, None)
        cam = (cam - cam.min())/(cam.max()-cam.min()+1e-8)
        cam_rs = cv2.resize(cam, (224,224))

        # Find peak activation
        y_peak, x_peak = np.unravel_index(cam_rs.argmax(), cam_rs.shape)
        x1 = max(0, x_peak - box_size//2)
        y1 = max(0, y_peak - box_size//2)
        x2 = min(224, x1 + box_size)
        y2 = min(224, y1 + box_size)

        # Draw box and label
        color = class_colors[cls]
        prob  = probs_dict[cls]
        cv2.rectangle(boxed, (x1,y1),(x2,y2), color, 2)
        text = f"{cls}: {prob:.2f}"
        (tw,th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(boxed, (x1, y1-th-5),(x1+tw, y1), color, -1)
        cv2.putText(boxed, text, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)


    fh.remove()
    bh.remove()

    boxed_path   = os.path.join(CONFIG["image_save_dir"], f"boxes_{uuid.uuid4().hex}.png")
    cv2.imwrite(boxed_path, boxed[..., ::-1])

    return {
        "filtered_symptoms": clean_text,
        "predicted_labels": labels,
        "predicted_probs":  probs_dict,
        "boxed_path":       boxed_path
    }


"""
# 1. Инициализация
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
policy_model = SymptomPolicyCNN(tokenizer.vocab_size, CONFIG["embed_dim"]).to(CONFIG["device"])
policy_ckpt = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
policy_model.load_state_dict(policy_ckpt["policy_model"])
policy_model.eval()

threshold = CONFIG["policy_threshold"]
device = CONFIG["device"]

# 2. Тестовые примеры
test_cases = [
    "",  # Пустая строка
    "[NO_SYMPTOMS_PROVIDED]", 
    "cough, fever, headache", 
    "I like Vilnius, blue sky", 
    "fever, blru sky, sroe throat",  
]

for idx, text in enumerate(test_cases):
    print(f"\n=== Test case {idx+1}: {repr(text)} ===")
    filtered, acts, bools = filter_symptom_phrases_with_policy(
        text, tokenizer, policy_model, threshold, device
    )
    print(f"Filtered symptoms: {filtered}")
    print(f"Acts: {acts}")
    print(f"Bools: {bools}")
"""

def load_images(image_paths):

    images = []
    if not image_paths:
        white = Image.fromarray(np.full((224, 224, 3), 255, dtype=np.uint8))
        black = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        return [(white, 'white'), (black, 'black')]
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            images.append((img, os.path.basename(path)))
        except Exception as e:
            print(f"Не удалось загрузить {path}: {e}")
    return images

"""def main():
    # Загрузить пользовательские картинки
    image_list = load_images(IMAGE_PATHS)

    # Заранее загрузить модели!
    models_symptom = load_models_symptom()

    # Пустые строки симптомов
    test_texts = [
        "",                             # настоящая пустая строка
        "[NO_SYMPTOMS_PROVIDED]",       # токен отсутствующих симптомов
    ]

    for image, name in image_list:
        for text in test_texts:
            print(f"\n--- Testing '{name}' with text: {repr(text)} ---")
            try:
                out = predict(
                    image_input=image,
                    symptoms_text=text,
                    label_names=LABEL_NAMES,
                    box_size=50,
                    model_dict=models_symptom  
                )
                print("Filtered symptoms:", out["filtered_symptoms"])
                print("Predicted labels & probs:", out["predicted_probs"])
                print("Boxes saved to:", out["boxed_path"])
            #    print("Heatmap saved to:", out["heatmap_path"])
            except Exception as e:
                print("Ошибка при Grad-CAM тесте:", e)


if __name__ == "__main__":
    main()
"""