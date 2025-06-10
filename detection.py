import torch
import yaml
import numpy as np
import cv2
from PIL import Image
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import platform

DATA_YAML = 'data.yaml'
MODEL_PATH = 'efficientdet_weapon_detection.pth'

with open(DATA_YAML) as f:
    data = yaml.safe_load(f)
    CLASS_NAMES = data['names']
    NUM_CLASSES = data['nc']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def play_alert():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)
    else:
        os.system('play -nq -t alsa synth 0.5 sine 1000')


def create_model(num_classes, image_size=512):
    config = get_efficientdet_config('efficientdet_d0')
    config.image_size = (image_size, image_size)
    config.norm_kwargs = dict(eps=0.001, momentum=0.01)
    config.num_classes = num_classes
    model = EfficientDet(config, pretrained_backbone=False)
    model.head = HeadNet(config, num_outputs=num_classes)
    return DetBenchPredict(model)


model = create_model(NUM_CLASSES)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.to(DEVICE)
model.eval()


transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


def predict_weapon(image_path):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    transformed = transform(image=img_np)
    img_tensor = transformed['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        detections = model(img_tensor)[0].detach().cpu().numpy()

    boxes, scores, labels = detections[:, :4], detections[:, 4], detections[:, 5].astype(int)
    mask = scores > 0.3
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    detected_labels = set()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label_name = CLASS_NAMES[label]
        detected_labels.add(label_name)
        cv2.putText(img_bgr, f"{label_name}: {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = f"output_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, img_bgr)

    if detected_labels:
        play_alert()
        return ', '.join(detected_labels) + f" detected {output_path}"
    else:
        return f"No weapon detected {output_path}"
