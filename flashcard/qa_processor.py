import os
import json
import tqdm
from concurrent.futures import ThreadPoolExecutor
import config

class QAProcessor:
    def __init__(self, qa_base_dir, qa_tocompare_dir):
        self.qa_base_dir = qa_base_dir
        self.qa_tocompare_dir = qa_tocompare_dir
        self.qa_data = {}

        """
        使用する質問と回答データを準備する
        """
        for base_image_path in base_image_paths:
            base_image_id = os.path.splitext(os.path.basename(base_image_path))[0].split('_')[-1]
            qa_base_path = os.path.join(self.qa_base_dir, f'qa_{base_image_id}.json')
            qa_tocompare_path = os.path.join(self.qa_tocompare_dir, f'qa_{base_image_id}.json')
            
            question, ans1, ans2 = self.get_qa_data(qa_base_path)
            _, ans1_compare, ans2_compare = self.get_qa_data(qa_tocompare_path)

            self.qa_data[base_image_id] = {
                "question": question,
                "ans1_base": ans1,
                "ans2_base": ans2,
                "ans1_compare": ans1_compare,
                "ans2_compare": ans2_compare
            }

    def _load_json(self, file_path):
        """
        Load a JSON file and return its contents.
        """
        with open(file_path, 'r') as file:
            return json.load(file)
    
    def get_qa_data(self, qa_file_path):
        qa_data = self._load_json(qa_file_path)
        return qa_data['question'], qa_data['ans1'], qa_data['ans2']
    
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

    def is_filter_passed(self, image_id, ans1_base, ans2_base, ans1_tocompare, ans2_tocompare):
        """特定の画像がフィルタ条件を満たすか判定する関数"""
        return (ans1_base['ans'][0] != ans2_base['ans'][0] and
                ans1_tocompare['ans'][0] != ans2_tocompare['ans'][0] and
                ans1_base['val'][0] >= self.filter_rate and
                ans2_base['val'][0] >= self.filter_rate and
                self.analysis_data[image_id]['base'] > self.analysis_data[image_id]['tocompare'])