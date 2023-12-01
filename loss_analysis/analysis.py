import numpy as np
import torchvision.models as models
import torch.nn.functional as F
import torch
import json
from tqdm import tqdm
import os
import config
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

class Analysis():
    def __init__(self, metrics_types, device):
        self.device = device
        self.metrics_types = metrics_types
        self.perceptual_loss = {}
        self.l1_norm_data = {}
        self.results = {metric: {'base': self._init_metric_categories(),
                                 'tocompare': self._init_metric_categories()}
                        for metric in metrics_types}
        self.loss_data = {}
        self.model = models.vgg19(pretrained=True).features.to(device)  # Example using VGG19
        self.model.eval()  # Set model to evaluation mode

    def _init_metric_categories(self):
        """ Initialize categories for each metric type """
        return {'all': [], 'color': [], 'shape': [],
                'diff_all': [], 'diff_color': [], 'diff_shape': [],
                'same_all': [], 'same_color': [], 'same_shape': []}

    def find_missing_analysis_types(self, existing_data):
        if not existing_data:
            return config.METRICS_TYPE

        missing_types = []
        for analysis_type in config.METRICS_TYPE:
            if not all(analysis_type in data for img_data in existing_data.values() for data in img_data.values()):
                missing_types.append(analysis_type)

        return missing_types


    def check_and_process_loss(self, loader, analysis_types):
        if os.path.exists(config.CALCULATED_LOSS_RESULT_PATH):
            print(f"Loading data from {config.CALCULATED_LOSS_RESULT_PATH}...")
            with open(config.CALCULATED_LOSS_RESULT_PATH, 'r') as json_file:
                self.loss_data = json.load(json_file)

            missing_types = self.find_missing_analysis_types(self.loss_data)

            if not missing_types:
                print("All analysis types are complete.")
                return self.loss_data
            else:
                print(f"Missing analysis types: {', '.join(missing_types)}")
        else:
            missing_types = [analysis_types]

        # 不足しているデータの計算
        for analysis_type in missing_types:
            self.process_batches(loader, analysis_type)

        # 結果を JSON に保存
        with open(config.CALCULATED_LOSS_RESULT_PATH, 'w') as json_file:
            json.dump(self.loss_data, json_file, indent=4)
        print("Updated data have been processed.")

        return self.loss_data
    
    def process_single_batch_perceptual_loss(self, batch_data):
        original_img, base_img, tocompare_img, img_ids = batch_data
        original_img = original_img.to(self.device)
        base_img = base_img.to(self.device)
        tocompare_img = tocompare_img.to(self.device)

        with torch.no_grad():
            for o_img, b_img, c_img, img_id in tqdm(zip(original_img, base_img, tocompare_img, img_ids), total=len(original_img), desc= "Calcurating perceptual loss...", leave=False):
                if img_id not in self.loss_data:
                    self.loss_data[img_id] = {'base': {}, 'tocompare': {}}

                # perceptual_loss の計算
                base_loss = self.calculate_perceptual_loss(o_img, b_img)
                tocompare_loss = self.calculate_perceptual_loss(o_img, c_img)

                # 結果を保存
                self.loss_data[img_id]['base']['perceptual_loss'] = base_loss.cpu().item()
                self.loss_data[img_id]['tocompare']['perceptual_loss'] = tocompare_loss.cpu().item()

    def process_single_batch_l1_norm(self, batch_data):
        original_img, base_img, tocompare_img, img_ids = batch_data
        original_img = original_img.to(self.device)
        base_img = base_img.to(self.device)
        tocompare_img = tocompare_img.to(self.device)

        with torch.no_grad():
            for o_img, b_img, c_img, img_id in tqdm(zip(original_img, base_img, tocompare_img, img_ids), total=len(original_img), desc= "Calcurating l1 norm...", leave=False):
                if img_id not in self.loss_data:
                    self.loss_data[img_id] = {'base': {}, 'tocompare': {}}

                # l1_norm の計算
                base_norm = self.calculate_l1_norm(o_img, b_img)
                tocompare_norm = self.calculate_l1_norm(o_img, c_img)

                # 結果を保存
                self.loss_data[img_id]['base']['l1_norm'] = base_norm.cpu().item()
                self.loss_data[img_id]['tocompare']['l1_norm'] = tocompare_norm.cpu().item()

    def process_batches(self, loader, analysis_type):
        print(f"Starting batch processing for {analysis_type}...")
        if analysis_type == 'perceptual_loss':
            process_function = self.process_single_batch_perceptual_loss
        elif analysis_type == 'l1_norm':
            process_function = self.process_single_batch_l1_norm
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        with ThreadPoolExecutor(max_workers=config.NUM_OF_WORKERS) as executor:
            futures = [executor.submit(process_function, batch_data) for batch_data in loader]
        #     # 各バッチの処理完了を追跡し、進捗バーを更新
        for future in as_completed(futures):
            pass
        print(f"{analysis_type.capitalize()} data have been processed.")

    # def process_qa_data(self, qa_data, metrics_data):
    #     question = qa_data['question']
    #     is_color_question = question.startswith("What color")
    #     is_shape_question = question.startswith("What shape")
    #     is_diff_answer = qa_data['ans1']['ans'][0] != qa_data['ans2']['ans'][0]

        # for metric, metric_value in metrics_data.items():
        #     self.results[metric]['all'].append(metric_value)
        #     if is_color_question:
        #         self.results[metric]['color'].append(metric_value)
        #     elif is_shape_question:
        #         self.results[metric]['shape'].append(metric_value)

        #     if is_diff_answer:
        #         self.results[metric]['diff_all'].append(metric_value)
        #         if is_color_question:
        #             self.results[metric]['diff_color'].append(metric_value)
        #         elif is_shape_question:
        #             self.results[metric]['diff_shape'].append(metric_value)

        #     elif not is_diff_answer:
        #         self.results[metric]['same_all'].append(metric_value)
        #         if is_color_question:
        #             self.results[metric]['same_color'].append(metric_value)
        #         elif is_shape_question:
        #             self.results[metric]['same_shape'].append(metric_value)

        # for metric in self.metrics_types:
        #     # Check if 'base' and 'tocompare' keys exist and if the specific metric is present
        #     base_metric_value = metrics_data.get('base', {}).get(metric)
        #     tocompare_metric_value = metrics_data.get('tocompare', {}).get(metric)

        #     # Process base metrics
        #     if base_metric_value is not None:
        #         self.results[metric]['all'].append(base_metric_value)
        #         if is_color_question:
        #             self.results[metric]['color'].append(base_metric_value)
        #         elif is_shape_question:
        #             self.results[metric]['shape'].append(base_metric_value)

        #         if is_diff_answer:
        #             self.results[metric]['diff_all'].append(base_metric_value)
        #             if is_color_question:
        #                 self.results[metric]['diff_color'].append(base_metric_value)
        #             elif is_shape_question:
        #                 self.results[metric]['diff_shape'].append(base_metric_value)
        #         else:
        #             self.results[metric]['same_all'].append(base_metric_value)
        #             if is_color_question:
        #                 self.results[metric]['same_color'].append(base_metric_value)
        #             elif is_shape_question:
        #                 self.results[metric]['same_shape'].append(base_metric_value)  
        # 
    def process_qa_data(self, qa_data, metrics_data):
        """
        Processes the QA data and updates the analysis results.

        :param qa_data: The QA data for a specific image.
        :param metrics_data: The metrics data containing 'base' and 'tocompare' keys.
        """
        question = qa_data['question']
        is_color_question = question.startswith("What color")
        is_shape_question = question.startswith("What shape")
        is_diff_answer = qa_data['ans1']['ans'][0] != qa_data['ans2']['ans'][0]

        for metric in self.metrics_types:
            if 'base' in metrics_data and metric in metrics_data['base']:
                self._update_results('base', metric, metrics_data['base'][metric],
                                     is_color_question, is_shape_question, is_diff_answer)
            if 'tocompare' in metrics_data and metric in metrics_data['tocompare']:
                self._update_results('tocompare', metric, metrics_data['tocompare'][metric],
                                     is_color_question, is_shape_question, is_diff_answer)

    def _update_results(self, source, metric, metric_value, is_color_question, is_shape_question, is_diff_answer):
        """ Update results for a specific metric source ('base' or 'tocompare') """
        categories = self._determine_categories(is_color_question, is_shape_question, is_diff_answer)
        for category in categories:
            self.results[metric][source][category].append(metric_value)

    def _determine_categories(self, is_color_question, is_shape_question, is_diff_answer):
        """ Determine the categories based on question type and answer difference """
        categories = ['all']
        if is_color_question:
            categories.append('color')
        elif is_shape_question:
            categories.append('shape')

        if is_diff_answer:
            categories.extend(['diff_all', 'diff_color' if is_color_question else 'diff_shape'])
        else:
            categories.extend(['same_all', 'same_color' if is_color_question else 'same_shape'])
        return categories
    
    def calculate_l1_norm(self, img1, img2, size=(256,256), img_channel=3):
        # Calculate L1 norm between img1 and img2
        l1_loss = torch.abs(img1 - img2).sum(dim=[0, 1, 2])  # バッチ内の各ペアについてL1ノルムを計算

        # L1ノルムを取りうる最大値で、正規化
        max_l1_norm = (255 * img_channel) * size[0] * size[1]
        # おそらくTimoの実装方法　計算
        normalize = img_channel * size[0] * size[1]
        l1_norm = l1_loss / normalize # たぶんmax_l1_normが正しい？
        return l1_norm

    def calculate_perceptual_loss(self, img1, img2):
        # Extract features using the pre-trained model
        features1 = self.model(img1)
        features2 = self.model(img2)
        # Calculate and return the Perceptual Loss
        return F.mse_loss(features1, features2)
    
    def calculate_mean_and_variance(self, values):
        # import pdb; pdb.set_trace()
        """指定された値のリストから平均と分散を計算する関数"""
        if values:  # リストが空でない場合に計算
            mean = format(np.mean(values), '.6f')  # 平均を小数点以下4桁の固定小数点表記でフォーマット
            std = format(np.std(values), '.6f')  # 標準偏差を小数点以下4桁の固定小数点表記でフォーマット
            return {'mean': mean, 'std': std}
        else:  # リストが空の場合、平均と分散は定義できない
            return {'mean': None, 'std': None}
        
    # def analyze_and_save_results(self, file_name):
    #     final_results = {}
    #     for metric, data in self.results.items():
    #         final_results[metric] = {}
    #         for category, values in data.items():
    #             final_results[metric][category] = self.calculate_mean_and_variance(values)

    #     with open(file_name, 'w') as f:
    #         json.dump(final_results, f, indent=4)
            
    def analyze_and_save_results(self, file_name):
        final_results = {}
        for metric, sources in self.results.items():
            final_results[metric] = {}
            for source, categories in sources.items():
                final_results[metric][source] = {}
                for category, values in categories.items():
                    final_results[metric][source][category] = self.calculate_mean_and_variance(values)
        
        # Save the updated structure
        with open(file_name, 'w') as f:
            json.dump(final_results, f, indent=4)