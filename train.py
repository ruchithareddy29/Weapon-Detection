from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from albumentations.pytorch import ToTensorV2
import albumentations as A
import yaml
from pycocotools.coco import COCO
from tqdm import tqdm

with open('data.yaml') as f:
    data = yaml.safe_load(f)
    CLASS_NAMES = data['names']
    NUM_CLASSES = data['nc']


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


class WeaponDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.coco = COCO(annotation_file)
        self.image_ids = self.coco.getImgIds()
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # Pascal VOC format
            labels.append(ann['category_id'])

        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['class_labels'], dtype=torch.int64)

        target = {'bbox': boxes, 'cls': labels}
        return image, target

# Create model
def create_model(num_classes):
    config = get_efficientdet_config('efficientdet_d0')
    config.num_classes = num_classes
    config.image_size = (512, 512)
    config.norm_kwargs = dict(eps=1e-3, momentum=0.01)

    net = EfficientDet(config, pretrained_backbone=True)
    net.head = HeadNet(config, num_outputs=num_classes)
    return DetBenchTrain(net, config)

# Initialize
dataset = WeaponDataset('data/images', 'data/annotations.json', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

model = create_model(NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


EPOCHS = 10
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, targets in progress:
        images = torch.stack(images).to(DEVICE)
        boxes = [t['bbox'].to(DEVICE) for t in targets]
        labels = [t['cls'].to(DEVICE) for t in targets]
        targets_formatted = [{'bbox': b, 'cls': l} for b, l in zip(boxes, labels)]

        loss_dict = model(images, targets_formatted)
        loss = loss_dict['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss:.4f}")

torch.save({'model_state_dict': model.model.state_dict()}, 'efficientdet_weapon_detection.pth')
print(" Model saved as efficientdet_weapon_detection.pth")



