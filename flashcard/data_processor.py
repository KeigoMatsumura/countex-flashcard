from PIL import Image
import cv2
import numpy as np
import os
import sys
import torch
from torchvision import transforms
import json
import tqdm
from concurrent.futures import ThreadPoolExecutor
import config

class DataProcessor:
    def __init__(self, original_img_dir, gen_img_base_dir, gen_img_tocompare_dir, qa_base_dir, qa_tocompare_dir, flashcard_save_img_dir):
        self.original_img_dir = original_img_dir
        self.gen_img_base_dir = gen_img_base_dir
        self.gen_img_tocompare_dir = gen_img_tocompare_dir
        self.flashcard_save_img_dir = flashcard_save_img_dir
        self.qa_base_dir = qa_base_dir
        self.qa_tocompare_dir = qa_tocompare_dir
        self.qa_data = {}
        self.img_data = {}
        self.size = (256,256)
        self.filter_rate = 0.8

        # 画像データの準備
        self.prepare_image_data()
        # QAデータの準備
        self.prepare_qa_data()

    def check_directory(self, directory):
        """
        指定されたディレクトリが存在するか確認し、存在しない場合はエラーを出力して終了する。
        """
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            sys.exit(1)

    def create_directory(self, directory):
        """
        指定されたディレクトリが存在するか確認し、存在しない場合は作成する。
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created: {directory}")

    def prepare_image_data(self):
        self.create_directory(config.FLASHCARD_CACHE_PATH)
        cache_path = os.path.join(config.FLASHCARD_CACHE_PATH, 'image_data_cache.json')

        if config.ENABLE_CACHE:
            if not config.REGENERATE_CACHE and os.path.exists(cache_path):
                with open(cache_path, 'r') as cache_file:
                    self.img_data = json.load(cache_file)
                return

        # キャッシュが無効、または再生成が必要な場合、画像データを処理
        self.img_data = self.process_image_data_parallel()

        if config.ENABLE_CACHE:
            with open(cache_path, 'w') as cache_file:
                json.dump(self.img_data, cache_file)

    def process_image_data_parallel(self):
        # 並列処理で画像データを処理する
        # ここではThreadPoolExecutorを使用
        with ThreadPoolExecutor() as executor:
            # 並列処理のタスクを定義（例：画像パスの取得）
            tasks = [executor.submit(self.get_image_paths, self.gen_img_base_dir, 'gen_im_'),
                     executor.submit(self.get_image_paths, self.gen_img_tocompare_dir, 'gen_im_')]

            # 並列処理の結果を受け取る
            base_image_paths = tasks[0].result()
            tocompare_image_paths = tasks[1].result()

        # 残りの処理
        self.original_image_paths = [self.get_original_image_path(os.path.splitext(os.path.basename(p))[0].split('_')[-1]) for p in base_image_paths]

        return {
            "img_ids": [os.path.splitext(os.path.basename(p))[0].split('_')[-1] for p in base_image_paths],
            "orig_img_paths": self.original_image_paths,
            "base_img_paths": base_image_paths,
            "tocompare_img_paths": tocompare_image_paths
        }
    
    def prepare_qa_data(self):
        self.create_directory(config.FLASHCARD_CACHE_PATH)
        cache_path = os.path.join(config.FLASHCARD_CACHE_PATH, 'qa_data_cache.json')

        if config.ENABLE_CACHE:
            if not config.REGENERATE_CACHE and os.path.exists(cache_path):
                with open(cache_path, 'r') as cache_file:
                    self.qa_data = json.load(cache_file)
                return

        # キャッシュが無効、または再生成が必要な場合、QAデータを処理
        self.qa_data = self.process_qa_data_parallel()
        # import pdb; pdb.set_trace()

        if config.ENABLE_CACHE:
            with open(cache_path, 'w') as cache_file:
                json.dump(self.qa_data, cache_file)
                
    def process_qa_data_parallel(self):
        with ThreadPoolExecutor() as executor:
            # 各画像IDに対してbaseとtocompareのQAデータを取得
            tasks = []
            for image_id in self.img_data['img_ids']:
                base_task = executor.submit(self.get_qa_data, os.path.join(self.qa_base_dir, f'qa_{image_id}.json'))
                tocompare_task = executor.submit(self.get_qa_data, os.path.join(self.qa_tocompare_dir, f'qa_{image_id}.json'))
                tasks.append((base_task, tocompare_task))

            # 並列処理の結果を受け取る
            qa_results = [(base_task.result(), tocompare_task.result()) for base_task, tocompare_task in tasks]

        # QAデータを整形して辞書に格納
        qa_data = {}
        for image_id, (base_qa, tocompare_qa) in zip(self.img_data['img_ids'], qa_results):
            question_base, ans1_base, ans2_base = base_qa
            _, ans1_tocompare, ans2_tocompare = tocompare_qa
            qa_data[image_id] = {
                "question": question_base,
                "ans1_base": ans1_base,
                "ans2_base": ans2_base,
                "ans1_tocompare": ans1_tocompare,
                "ans2_tocompare": ans2_tocompare
            }
        return qa_data

    def get_image_and_qa_data(self, index):
        image_id = self.img_data['img_ids'][index]
        original_image = Image.open(self.img_data['orig_img_paths'][index])
        base_image = Image.open(self.img_data['base_img_paths'][index])
        tocompare_image = Image.open(self.img_data['tocompare_img_paths'][index]) 

        original_image = self.resize_image(original_image, self.size)
        base_image = self.resize_image(base_image, self.size)
        tocompare_image = self.resize_image(tocompare_image, self.size)
        
        original_image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        base_img_cv = cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR)
        tocompare_img_cv = cv2.cvtColor(np.array(tocompare_image), cv2.COLOR_RGB2BGR)
        # import pdb; pdb.set_trace()
        images = {
            "img_id": image_id,
            "orig_img": original_image_cv,
            "base_img": base_img_cv,
            "tocompare_img": tocompare_img_cv
        }

        qas = self.qa_data[image_id]

        return images, qas
    
    @staticmethod
    def add_red_border(img_cv, border_size=10, color=(0, 0, 255)):
        height, width = img_cv.shape[:2]
        top, bottom, left, right = [border_size]*4
        img_with_border = cv2.rectangle(img_cv.copy(), (left - border_size, top - border_size),
                                        (width + right - border_size, height + bottom - border_size),
                                        color, border_size)
        return img_with_border
        
    def get_image_paths(self, directory, pattern):
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(pattern)]

    def resize_image(self, image, size):
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

    def _load_json(self, file_path):
        """
        Load a JSON file and return its contents.
        """
        with open(file_path, 'r') as file:
            return json.load(file)
    
    def is_json(self, fpath):
        try:
            with open(fpath, 'r') as f:
                json.load(f)
            return True
        except json.JSONDecodeError:
            return False

    def convert_file_to_json(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        json_data = {
            "question": lines[0].strip().split("question:")[1].strip(),
            "ans1": json.loads(lines[1].strip()),
            "ans2": json.loads(lines[2].strip())
        }
        json_file_path = os.path.splitext(file_path)[0] + ".json"
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        os.remove(file_path)  # Delete the original text file

    def all_files_are_json(self, directory):
        if not os.path.exists(directory):
            return False
        return all(file.endswith(".json") and self.is_json(os.path.join(directory, file)) for file in os.listdir(directory))


    def convert_txt_to_json(self, input_directory):
        txt_files = [f for f in os.listdir(input_directory) if f.endswith('.txt')]
        file_paths = [os.path.join(input_directory, f) for f in txt_files]
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(self.convert_file_to_json, file_paths), total=len(txt_files), desc="Converting TXT to JSON"))

    def process_qa_files(self, directory):
        if not os.path.exists(directory) or not os.listdir(directory):
            print(f"Directory does not exist or is empty: {directory}")
            return False

        print("Checking if all files are already JSON... This may take some time.")
        if self.all_files_are_json(directory):
            print(f"All files in {directory} are already JSON.")
            return True

        self.convert_txt_to_json(directory)

        if self.all_files_are_json(directory):
            print(f"All files in {directory} have been converted to JSON.")
            return True
        else:
            print(f"Failed to convert some files in {directory}.")
            return False
    
    def is_filter_passed(self, image_id, analized_data):
        """特定の画像がフィルタ条件を満たすか判定する関数"""
        qa_data = self.qa_data[image_id]
        return (qa_data['ans1_base']['ans'][0] != qa_data['ans2_base']['ans'][0] and
                qa_data['ans1_tocompare']['ans'][0] != qa_data['ans2_tocompare']['ans'][0] and
                qa_data['ans1_base']['val'][0] >= self.filter_rate and
                qa_data['ans2_base']['val'][0] >= self.filter_rate and
                analized_data[image_id]['base'] > analized_data[image_id]['tocompare'])