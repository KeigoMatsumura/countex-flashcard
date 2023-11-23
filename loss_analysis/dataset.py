import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# データセットクラスを定義
class ImagePairDataset(Dataset):
    def __init__(self, original_dir, base_dir, tocompare_dir):
        self.original_dir = original_dir
        self.base_dir = base_dir
        self.tocompare_dir = tocompare_dir
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 画像を256x256にリサイズ
            transforms.Grayscale(num_output_channels=3),  # すべての画像を3チャネルに変換
            transforms.ToTensor(),  # PIL ImageをPyTorchテンソルに変換
        ])
        self.image_ids = [filename.split("_")[-1].split(".")[0] for filename in os.listdir(base_dir) if filename.startswith("gen_im_") and filename.endswith(".png")]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        original_img_path = os.path.join(self.original_dir, f"COCO_train2014_{img_id.zfill(12)}.jpg")
        base_img_path = os.path.join(self.base_dir, f"gen_im_{img_id}.png")
        tocompare_img_path = os.path.join(self.tocompare_dir, f"gen_im_{img_id}.png")

        original_img = Image.open(original_img_path)
        base_img = Image.open(base_img_path)
        tocompare_img = Image.open(tocompare_img_path)

        original_img = self.transform(original_img)
        base_img = self.transform(base_img)
        tocompare_img = self.transform(tocompare_img)

        return original_img, base_img, tocompare_img, img_id
