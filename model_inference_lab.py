import os
import uuid
import io
import re
from difflib import get_close_matches
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import  transforms
from timm import create_model
import matplotlib.pyplot as plt

# === CONFIG ===
CONFIG = {
    "model_path": "best_model_labs.pth",
    "artifacts_path": "lab_artifacts.pth", 
    "image_save_dir": "results",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "cam_threshold": 0.2
}
os.makedirs(CONFIG["image_save_dir"], exist_ok=True)


class LabImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.backbone = create_model(
            'seresnet50',
            pretrained=True,
            features_only=False
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.projection_head = self._init_embedding_layer(output_dim)

    def _init_embedding_layer(self, output_dim):

        in_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)
        return nn.Linear(in_features, output_dim)

    def forward(self, x, return_features=False):
        conv_features = self.backbone.forward_features(x) 
        global_avg = conv_features.mean(dim=[2, 3])  
        image_projection = self.projection_head(global_avg) 
        image_embedding = F.normalize(image_projection, dim=-1) 

        if return_features:
            return image_embedding, conv_features
        return image_embedding

class LabEncoder(nn.Module):
    def __init__(self, test_name_embeddings, hidden_dim=512, fusion_mode=None, use_attention=True):

        super().__init__()
        self.embedding_dim = test_name_embeddings.size(1)
        self.fusion_mode = fusion_mode
        self.use_attention = use_attention
        self.test_name_embedding = nn.Embedding.from_pretrained(test_name_embeddings, freeze=False)
        self.value_proj = nn.Linear(1, self.embedding_dim)
        self.gate_proj = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        def gated_fusion(test, val):
            gate_input = torch.cat([test, val], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_input))
            return gate * test + (1 - gate) * val
        self.fusion_fn = gated_fusion

        if use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=4, batch_first=True)
        self.output_proj = nn.Linear(self.embedding_dim, hidden_dim)

    def forward(self, test_ids, test_values, mask=None):
        test_embs = self.test_name_embedding(test_ids)                    # [B, T, D]
        value_embs = self.value_proj(test_values.unsqueeze(-1))           # [B, T, D]
        combined = self.fusion_fn(test_embs, value_embs)                  # [B, T, D]

        if self.use_attention:
            attention_mask = ~mask if mask is not None else None
            attn_output, _ = self.attention(combined, combined, combined, key_padding_mask=attention_mask)
            combined = attn_output                                       # [B, T, D]

        output = self.output_proj(combined)                               # [B, T, hidden_dim]
        return output

class Attention(nn.Module):

    def __init__(self, embed_dim, num_lab_tests, heads=4, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head
        self.to_q = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(0.1)
        )

        self.lab_test_bias = nn.Embedding(num_lab_tests + 1, 1)
        self.value_proj = nn.Linear(1, 1)
        self.joint_fusion_proj = nn.Sequential(nn.Linear(2, 1), nn.Tanh())

    def forward(
        self,
        image_embeddings, 
        lab_embeddings=None,
        mask=None,   
        lab_test_indices=None,  
        lab_values=None  
    ):

        if lab_embeddings is None:
            lab_embeddings = image_embeddings  

        q = self.to_q(image_embeddings)
        k = self.to_k(lab_embeddings)
        v = self.to_v(lab_embeddings)

        B, Q_len, _ = q.shape
        _, K_len, _ = k.shape

        q = q.view(B, Q_len, self.heads, -1).transpose(1, 2) 
        k = k.view(B, K_len, self.heads, -1).transpose(1, 2) 
        v = v.view(B, K_len, self.heads, -1).transpose(1, 2) 

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  

        if lab_test_indices is not None and lab_values is not None:
            test_bias = self.lab_test_bias(lab_test_indices).squeeze(-1)
            value_bias = self.value_proj(lab_values.unsqueeze(-1)).squeeze(-1)  

            joint_input = torch.stack([test_bias, value_bias], dim=-1)
            test_value_bias = self.joint_fusion_proj(joint_input).squeeze(-1) 
            test_value_bias = torch.clamp(test_value_bias, min=-5, max=5)

            test_value_bias = test_value_bias.unsqueeze(1).unsqueeze(2) 
            attention_scores = attention_scores + test_value_bias

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)    
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1) 
        attended = torch.matmul(attention_weights, v)  
        attended = attended.transpose(1, 2).contiguous().view(B, Q_len, -1)  

        fused = self.to_out(attended)  
        output = fused + image_embeddings  

        return output


class CrossAttentionFusion(nn.Module):

    def __init__(self, embed_dim, num_lab_tests):
        super().__init__()
        self.attention = Attention(embed_dim=embed_dim, num_lab_tests=num_lab_tests, heads=4, dim_head=64)

    def forward(
        self,
        image_embedding, 
        lab_embeddings, 
        mask=None,  
        lab_test_indices=None,  
        lab_values=None 
    ):

        if image_embedding.dim() == 2:
            image_embedding = image_embedding.unsqueeze(1) 

        if mask is not None:
            no_lab_data = mask.sum(dim=1) == 0 
            if no_lab_data.any():

                fused_output = torch.zeros_like(image_embedding)

                has_lab_indices = (~no_lab_data).nonzero(as_tuple=True)[0]
                if len(has_lab_indices) > 0:
                    fused_output[has_lab_indices] = self.attention(
                        image_embeddings=image_embedding[has_lab_indices],
                        lab_embeddings=lab_embeddings[has_lab_indices],
                        mask=mask[has_lab_indices],
                        lab_test_indices=lab_test_indices[has_lab_indices],
                        lab_values=lab_values[has_lab_indices]
                    )

                no_lab_indices = no_lab_data.nonzero(as_tuple=True)[0]
                fused_output[no_lab_indices] = image_embedding[no_lab_indices]
                return fused_output

        return self.attention(
            image_embeddings=image_embedding,
            lab_embeddings=lab_embeddings,
            mask=mask,
            lab_test_indices=lab_test_indices,
            lab_values=lab_values
        )

class MultimodalClassifierSCAN(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, fused_sequence):

        if fused_sequence.dim() == 2:
            fused_sequence = fused_sequence.unsqueeze(1)  
        λ = 2.0

        pooled = (1/λ) * torch.logsumexp(λ * fused_sequence, dim=1)  
        logits = self.classifier(pooled)  
        return logits

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_models_lab():
    # 1) Загружаем артефакты
    art = torch.load(
        CONFIG['artifacts_path'],
        map_location=CONFIG['device'],
        weights_only=False
    )
    stats   = art['test_statistics']
    idx_map = art['test_name_to_idx']    
    embs    = art['test_name_embeddings']   

    # 3) Достаём остальные параметры
    names         = art['label_list']
    th            = art.get('best_thresholds', None)
    hidden_dim    = art['hidden_dim']
    fusion_mode   = art['fusion_mode']
    use_attention = art['use_attention']

    # 4) Создаём модули, передаём embs в LabEncoder
    img_enc = LabImageEncoder(output_dim=hidden_dim).to(CONFIG['device'])
    lab_enc = LabEncoder(
        test_name_embeddings=embs,
        hidden_dim=hidden_dim,
        fusion_mode=fusion_mode,
        use_attention=use_attention
    ).to(CONFIG['device'])
    xatt = CrossAttentionFusion(
        embed_dim=hidden_dim,
        num_lab_tests=len(idx_map)
    ).to(CONFIG['device'])
    clf = MultimodalClassifierSCAN(
        embed_dim=hidden_dim,
        num_classes=len(names)
    ).to(CONFIG['device'])

    # 5) Загружаем веса из лучшей модели
    st = torch.load(
        CONFIG['model_path'],
        map_location=CONFIG['device'],
        weights_only=False
    )
    lab_enc.load_state_dict(st['lab_encoder'])
    img_enc.load_state_dict(st['image_encoder'])
    if 'cross_attention' in st:
        xatt.load_state_dict(st['cross_attention'])
    clf.load_state_dict(st['classifier'])
    th = st.get('best_thresholds', th)

    # 6) Переключаем в eval
    for m in (lab_enc, img_enc, xatt, clf):
        m.eval()

    # 7) Возвращаем всё
    return {
        'stats':   stats,
        'idx_map': idx_map,
        'names':   names,
        'th':      th,
        'img_enc': img_enc,
        'lab_enc': lab_enc,
        'xatt':    xatt,
        'clf':     clf
    }
from store_models import models
def predict(image_input, lab_dict, cam_per_class=True):

    mdl = models['lab']
    if isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input)).convert('RGB')
    elif isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    else:
        img = image_input.convert('RGB')
    inp = img_transform(img).unsqueeze(0).to(CONFIG['device']) 

    print(">>> Incoming lab_dict:", lab_dict)  
    ids, vals, mask, filtered_labs = preprocess_labs(lab_dict, mdl['stats'], mdl['idx_map'])
    print(">>> After preprocess_labs →",
          "ids:", ids.tolist(),
          "vals:", vals.tolist(),
          "mask:", mask.tolist())   

    conv_feats, grads = None, None
    def forward_hook(module, inp, out):
        nonlocal conv_feats
        conv_feats = out
        conv_feats.requires_grad_(True)
        conv_feats.retain_grad()

    def backward_hook(module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0]

    layer4 = mdl['img_enc'].backbone.layer4
    fh = layer4.register_forward_hook(forward_hook)
    bh = layer4.register_full_backward_hook(backward_hook)

    # 5) Прямой проход
    img_emb = mdl['img_enc'](inp)
    lab_emb = mdl['lab_enc'](ids, vals, mask)
    fused   = mdl['xatt'](img_emb, lab_emb, mask, ids, vals)
    logits  = mdl['clf'](fused).squeeze(0)
    probs   = torch.sigmoid(logits).detach().cpu().numpy()
    pred    = (probs >= mdl['th']).astype(int)

    # 6) Списки меток и вероятностей
    labels     = [n for n,p in zip(mdl['names'], pred) if p]
    probs_dict = {n: float(s) for n,s,p in zip(mdl['names'], probs, pred) if p}

    # Цвета для боксов
    cmap = plt.get_cmap('tab10')
    class_colors = {lbl: tuple((np.array(cmap(i))[:3]*255).astype(int).tolist())
                    for i,lbl in enumerate(labels)}

    # 7) Подготовка изображения для отрисовки
    raw_np = np.array(img.resize((224,224)))
    boxed  = raw_np.copy()

    # 8) По-классовый Grad-CAM + один бокс вокруг максимума
    box_size = 50  # размер бокса
    for cls in labels:
        cls_idx = mdl['names'].index(cls)

        mdl['clf'].zero_grad()
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

        # Находим пик активации
        y_peak, x_peak = np.unravel_index(cam_rs.argmax(), cam_rs.shape)
        x1 = max(0, x_peak - box_size//2)
        y1 = max(0, y_peak - box_size//2)
        x2 = min(224, x1 + box_size)
        y2 = min(224, y1 + box_size)

        # Рисуем бокс и подпись
        color = class_colors[cls]
        prob  = probs_dict[cls]
        cv2.rectangle(boxed, (x1,y1),(x2,y2), color, 2)
        txt = f"{cls}: {prob:.2f}"
        (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(boxed, (x1, y1-th-5),(x1+tw, y1), color, -1)
        cv2.putText(boxed, txt, (x1, y1-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # 9) Удаляем хуки
    fh.remove()
    bh.remove()

    # 10) Сохраняем
    heatmap_path = os.path.join(CONFIG['image_save_dir'], f"heatmap_{uuid.uuid4().hex}.png")
    boxed_path   = os.path.join(CONFIG['image_save_dir'], f"boxes_{uuid.uuid4().hex}.png")
    cv2.imwrite(boxed_path, boxed[..., ::-1])

    return {
        'predicted_labels': labels,
        'predicted_probs':  probs_dict,
        'heatmap_path':     heatmap_path,
        'boxed_path':       boxed_path,
        'filtered_labs':    filtered_labs
    }

def preprocess_labs(lab_dict, stats, idx_map, fuzzy_cutoff=0.8):

    seq = []
    dropped = []
    filtered_lab_entries = []    # <--- Новый список

    stats_keys = list(stats.keys())

    def normalize_name(name: str) -> str:
        n = name.strip().lower()
        n = re.sub(r"\([^)]*\)", "", n)
        n = re.sub(r"[^a-z0-9\s]", "", n)
        n = re.sub(r"\s+", " ", n)
        return n.strip()

    norm_map = {normalize_name(k): k for k in stats_keys}

    for raw_name, raw_value in lab_dict.items():
        norm = normalize_name(raw_name)
        if norm in norm_map:
            name = norm_map[norm]
            print(f"Input test '{raw_name}' matched exactly to '{name}'")
        else:
            match = get_close_matches(norm, norm_map.keys(), n=1, cutoff=fuzzy_cutoff)
            if match:
                name = norm_map[match[0]]
                print(f"Input test '{raw_name}' fuzzy-matched to '{name}'")
            else:
                dropped.append(raw_name)
                print(f"Input test '{raw_name}' is unknown and will be dropped")
                continue
        mean, std = stats[name]
        val_norm = (float(raw_value) - mean) / (std + 1e-6)
        seq.append((name, val_norm))

        filtered_lab_entries.append({"test": name, "value": raw_value})

    if not seq:
        seq = [("pad", 0.0)]
    if dropped:
        print(f"Dropping unknown tests: {dropped}")

    ids, vals, mask = [], [], []
    for name, val in seq:
        ids.append(idx_map.get(name, idx_map['pad']))
        vals.append(val)
        mask.append(1 if name != 'pad' else 0)

    # filtered_lab_entries пропускаем только 'pad'
    filtered_lab_entries = [lab for lab in filtered_lab_entries if lab["test"] != "pad"]

    return (
        torch.tensor([ids], dtype=torch.long, device=CONFIG['device']),
        torch.tensor([vals], dtype=torch.float, device=CONFIG['device']),
        torch.tensor([mask], dtype=torch.bool, device=CONFIG['device']),
        filtered_lab_entries   
    )


"""# === MAIN для тестирования ===
def main():
    img_path = "raw_files/effusion.png"
    lab_inputs = {
        "wbc": 123.4,
    #    "WBC":    7.2,
    #    "CRP (mg/L)":      250,
    #    "Proalcitonn": 13.5
    }

    result = predict(img_path, lab_inputs)
    print("Predicted labels:", result['predicted_labels'])
    print("Predicted probabilities:", result['predicted_probs'])
    print("Heatmap:", result['heatmap_path'])
    print("Boxes:", result['boxed_path'])

if __name__ == "__main__":
    main()"""
