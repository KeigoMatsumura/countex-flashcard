from PIL import Image
import cv2
import numpy as np
import os
import sys
import torch
from torchvision import transforms

class ImgProcessor:
    def __init__(self, original_img_dir, gen_img_base_dir, gen_img_tocompare_dir, flashcard_save_img_dir):
        self.original_img_dir = original_img_dir
        self.gen_img_base_dir = gen_img_base_dir
        self.gen_img_tocompare_dir = gen_img_tocompare_dir
        self.flashcard_save_img_dir = flashcard_save_img_dir
        self.img_data = {}
        """
        使用する画像データを準備する
        """
        # 元画像のディレクトリがない場合は修了
        if not os.path.exists(self.original_img_dir):
            print(f"Original Image directory does not exist: {self.original_img_dir}")
            sys.exit(1)
        # 比較する生成画像１のディレクトリがない場合修了
        if not os.path.exists(self.gen_img_base_dir):
            print(f"Base directory does not exist: {self.gen_img_base_dir}")
            sys.exit(1)
        # 比較する生成画像２のディレクトリがない場合は修了
        if not os.path.exists(self.gen_img_tocompare_dir):
            print(f"Comparison directory does not exist: {self.gen_img_tocompare_dir}")
            sys.exit(1)
        # 保存先に指定されたディレクトリが存在しない場合は作成
        if not os.path.exists(self.flashcard_save_img_dir):
            os.makedirs(self.flashcard_save_img_dir)
            print(f"Created directory: {self.flashcard_save_img_dir}")
        else:
            print(f"Directory already exists: {self.flashcard_save_img_dir}")
        
        # 生成画像１の画像パスリストを生成する
        # 比較する生成画像１のディレクトリ中から，’gen_im_’から始まるファイルを検索し，なければ修了
        base_image_paths = self.get_image_paths(self.gen_img_base_dir, 'gen_im_')
        if not base_image_paths:
            print(f"No images found in base directory: {self.gen_img_base_dir}")
            sys.exit(1)
        # 生成画像２の画像パスリストを生成する
        # 比較する生成画像１のディレクトリ中から，’gen_im_’から始まるファイルを検索し，なければ修了
        tocompare_image_paths = self.get_image_paths(self.gen_img_tocompare_dir, 'gen_im_')
        if not tocompare_image_paths:
            print(f"No images found in base directory: {self.gen_img_tocompare_dir}")
            sys.exit(1)
        # 元画像の画像パスリストを生成する
        # 元画像のディレクトリ中から，’png'か'jpg'の拡張子ファイルを検索し，なければ修了
        for base_image_path in base_image_paths:
            base_image_id = os.path.splitext(os.path.basename(base_image_path))[0].split('_')[-1]
            original_image_paths = self.get_original_image_path(base_image_id)

        self.img_data = {
            "img_id": base_image_id,
            "orig_img_paths": original_image_paths,
            "base_img_paths": base_image_paths,
            "tocompare_img_path": tocompare_image_paths
        }

    
    def add_red_border(img_cv, border_size=10, color=(0, 0, 255)):
        height, width = img_cv.shape[:2]
        top, bottom, left, right = [border_size]*4
        img_with_border = cv2.rectangle(img_cv.copy(), (left - border_size, top - border_size),
                                        (width + right - border_size, height + bottom - border_size),
                                        color, border_size)
        return img_with_border
        
    def get_image_paths(directory, pattern):
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(pattern)]

    def resize_image(image, size):
        # Check if image is a PIL Image
        if isinstance(image, Image.Image):
            current_size = image.size  # (width, height)
            if current_size != size:
                return image.resize(size, Image.ANTIALIAS)
            return image
        # Check if image is a numpy array (OpenCV image)
        elif isinstance(image, np.ndarray):
            # OpenCV size is in (height, width)
            current_size = image.shape[1], image.shape[0]  # (width, height)
            if current_size != size:
                return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            return image
        # Check if image is a numpy array (OpenCV image)
        elif isinstance(image, torch.Tensor):
            # リサイズを行う変換を定義
            resize_transform = transforms.Resize(size, antialias=True)
            # リサイズ変換を適用
            image = resize_transform(image)
            return image
        else:
            raise TypeError("Unsupported image type")
    
    def get_imgs(self, index):
        original_image = Image.open(self.img_data['orig_img_path'][index])
        base_image = Image.open(self.img_data['base_img_path'][index])
        tocomapre_image = Image.open(self.img_data['tocompare_img_path'][index]) 

        original_image = self.resize_image(original_image, self.size)
        base_image = self.resize_image(base_image, self.size)
        tocomapre_image = self.resize_image(tocomapre_image, self.size)
        
        original_image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        base_img_cv = cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR)
        tocomapre_image_cv = cv2.cvtColor(np.array(tocomapre_image), cv2.COLOR_RGB2BGR)

        cv_images = { 
            "orig_img": original_image_cv,
            "base_img": base_img_cv,
            "tcompare_img": tocomapre_image_cv
        }
        return cv_images

    def get_original_image_path(self, base_image_id):
        padding_needed = 12 - len(base_image_id)
        padded_id = '0' * padding_needed + base_image_id
        original_image_name = f'COCO_train2014_{padded_id}.jpg'
        return os.path.join(self.original_img_dir, original_image_name)
    
    # patternにマッチするファイル名の，完全なパスを全ての項目取り出し，リストとして返す
    def get_image_paths(self, directory, pattern):
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(pattern)]
    
    def get_qa_data(self, qa_file_path):
        qa_data = self._load_json(qa_file_path)
        return qa_data['question'], qa_data['ans1'], qa_data['ans2']
