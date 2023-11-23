import cv2
import os
import numpy as np
import config

class WindowGenerator:
    def generate(self, images, qas, analyzed_data):
        image_id = images['img_id']
    
        orig_img = self.add_text_above_image(images['orig_img'], "Original")
        base_img = self.add_text_above_image(images['base_img'], "w/Lxmert")
        tocompare_img = self.add_text_above_image(images['tocompare_img'], "w/Mutan")

        orig_img = self.add_key_instructions_below_image(orig_img, config.KEY_INSTRUCTIONS)
        base_img = self.add_answers_below_image(base_img, [("ANS1", qas['ans1_base']), ("ANS2", qas['ans2_base'])])
        tocompare_img = self.add_answers_below_image(tocompare_img, [("ANS1", qas['ans1_tocompare']), ("ANS2", qas['ans2_tocompare'])])

        base_analysis_data = {
            "Perceptual Loss": analyzed_data[image_id]['base']['perceptual_loss'],
            "L1 Norm": analyzed_data[image_id]['base']['l1_norm']
        }
        tocompare_analysis_data = {
            "Perceptual Loss": analyzed_data[image_id]['tocompare']['perceptual_loss'],
            "L1 Norm": analyzed_data[image_id]['tocompare']['l1_norm']
        }

        base_img = self.add_analysis_data_label_below_image(base_img, base_analysis_data)
        tocompare_img = self.add_analysis_data_label_below_image(tocompare_img, tocompare_analysis_data)

        # base_img = self.add_analysis_data_label_below_image(base_img, analized_data[image_id]['base'], label=config.METRICS_LABEL_NAME +":")
        # tocompare_img = self.add_analysis_data_label_below_image(tocompare_img, analized_data[image_id]['tocompare'], label=config.METRICS_LABEL_NAME +":")

        orig_img, base_img, tocompare_img = self.make_same_height(orig_img, base_img, tocompare_img)

        concatinated_image = cv2.hconcat([orig_img, base_img, tocompare_img])
        concatinated_image = self.add_question_and_imageid_above_images(concatinated_image, qas['question'], f"ID: {image_id}")
        return concatinated_image

    def resize_image(self, image):
        return self.resize_image(image, self.size)

    def convert_to_opencv(self, pil_image):
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def add_red_border(self, img_cv, border_size=10, color=(0, 0, 255)):
        height, width = img_cv.shape[:2]
        top, bottom, left, right = [border_size]*4
        img_with_border = cv2.rectangle(img_cv.copy(), (left - border_size, top - border_size),
                                        (width + right - border_size, height + bottom - border_size),
                                        color, border_size)
        return img_with_border

    def add_text_above_image(self, img_cv, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1, text_color=(0, 0, 0), background_color=(255, 255, 255)):
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_width, text_height = text_size
        extended_img = cv2.copyMakeBorder(img_cv, text_height * 2, 0, 0, 0, cv2.BORDER_CONSTANT, value=background_color)
        text_offset_x = (extended_img.shape[1] - text_width) // 2
        text_offset_y = text_height
        cv2.putText(extended_img, text, (text_offset_x, text_offset_y), font, font_scale, text_color, font_thickness)
        return extended_img
    
    def make_same_height(self, img1, img2, img3):
        max_height = max(img1.shape[0], img2.shape[0], img3.shape[0])

        def pad_image(img, height):
            if img.shape[0] < height:
                padding = height - img.shape[0]
                img = cv2.copyMakeBorder(img, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            return img

        img1 = pad_image(img1, max_height)
        img2 = pad_image(img2, max_height)
        img3 = pad_image(img3, max_height)

        return img1, img2, img3


    def check_image_exists(self, image_path):
        return os.path.exists(image_path)

    def get_original_image_path(self, base_image_id):
        padding_needed = 12 - len(base_image_id)
        padded_id = '0' * padding_needed + base_image_id
        original_image_name = f'COCO_train2014_{padded_id}.jpg'
        return os.path.join(config.ORIG_IMG_DIR, original_image_name)
    
    def add_key_instructions_below_image(self, img, instructions, font_scale=0.5, font_thickness=1, text_color=(0, 0, 0), left_margin=10):
        lines = [item.strip() for item in instructions.split('|')]
        total_text_height = 0
        line_heights = []
        for line in lines:
            (line_width, line_height), baseline = cv2.getTextSize(line.strip(), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            total_text_height += line_height * 2 + baseline
            line_heights.append(line_height + baseline)
        image_with_border = cv2.copyMakeBorder(img, 0, total_text_height + baseline * len(lines), 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        y_offset = img.shape[0] + baseline + line_heights[0]
        for line in lines:
            org = (left_margin, y_offset)
            cv2.putText(image_with_border, line, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
            y_offset += line_heights[lines.index(line)] + baseline

        return image_with_border

    def add_question_and_imageid_above_images(self, img_cv, question, image_id, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, font_thickness=1, text_color=(0, 0, 0)):
        question_text_size = cv2.getTextSize(question, font, font_scale, font_thickness)[0]
        image_id_text_size = cv2.getTextSize(image_id, font, font_scale, font_thickness)[0]
        padding_above_line = 10
        padding_between_lines = 5
        padding_below_line = 5
        # 余白の高さを計算（質問文と画像IDのため）
        total_text_height = question_text_size[1] + image_id_text_size[1] + padding_above_line + padding_between_lines + padding_below_line
        # 画像の上に余白を追加
        extended_img = cv2.copyMakeBorder(img_cv, total_text_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # 画像IDを描画
        image_id_x = (extended_img.shape[1] - image_id_text_size[0]) // 2  # 中央揃え
        image_id_y = padding_above_line + image_id_text_size[1]
        cv2.putText(extended_img, image_id, (image_id_x, image_id_y), font, font_scale, text_color, font_thickness)
        # 質問文を描画
        question_x = (extended_img.shape[1] - question_text_size[0]) // 2  # 中央揃え
        question_y = image_id_y + padding_between_lines + question_text_size[1]
        cv2.putText(extended_img, question, (question_x, question_y), font, font_scale, text_color, font_thickness)

        return extended_img
        
    # def add_analysis_data_label_below_image(self, img_cv, analysis_data_value, label="Perceptual Dist:", font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1, text_color=(0, 0, 0), start_x=3):
    #     # ラベルとL1ノルムの値を組み合わせてテキストを作成
    #     text = f"{label} {analysis_data_value:.4f}"  # 小数点以下4桁でフォーマット

    #     # テキストのサイズを取得
    #     text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    #     text_width, text_height = text_size

    #     # 画像の下に余白を追加
    #     extended_img = cv2.copyMakeBorder(img_cv, 0, text_height + 20, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    #     # テキストのY位置を計算（中央揃えではなく下部余白の上部に配置）
    #     text_offset_y = extended_img.shape[0] - text_height

    #     # テキストを画像の下に左揃えで配置
    #     cv2.putText(extended_img, text, (start_x, text_offset_y), font, font_scale, text_color, font_thickness)

    #     return extended_img

    def add_analysis_data_label_below_image(self, img_cv, analysis_data, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1, text_color=(0, 0, 0), start_x=3):
        """
        Add multiple analysis data labels below the image.

        analysis_data: A dictionary where keys are the label names (e.g., 'Perceptual Loss', 'L1 Norm') and values are the corresponding data values.
        """
        y_offset = img_cv.shape[0] + 10  # Starting Y position for the first label
        extended_img = img_cv.copy()

        for label, value in analysis_data.items():
            text = f"{label}: {value:.4f}"  # Format each label with its value
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_height = text_size[1]

            extended_img = cv2.copyMakeBorder(extended_img, 0, text_height + 10, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(extended_img, text, (start_x, y_offset), font, font_scale, text_color, font_thickness)
            y_offset += text_height + 5  # Update Y position for the next label

        return extended_img
    
    def add_answers_below_image(self, img, answers_with_values, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1, text_color=(0, 0, 0)):
        line_height = 20
        padding = 3
        out_padding = 20
        start_x = padding
        start_y = img.shape[0] + padding + line_height
        # This part handles the two types of arguments
        if all(isinstance(item, tuple) for item in answers_with_values):
            # When tuples are provided, we expect (source_name, answer_data) format
            total_additional_height = out_padding * 3 + (line_height + padding) * sum(len(source_with_data[1]['ans']) for source_with_data in answers_with_values)
        else:
            # When a single dictionary is provided
            total_additional_height = out_padding * 3 + (line_height + padding) * len(answers_with_values['ans'])
        # Add space below the image for text
        img_with_space_for_text = cv2.copyMakeBorder(img, 0, total_additional_height, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # Check if the input is a list of tuples or a single dictionary and handle accordingly
        if isinstance(answers_with_values, list) and all(isinstance(item, tuple) for item in answers_with_values):
            for source_name, answer_data in answers_with_values:
                # Display the source name
                cv2.putText(img_with_space_for_text, source_name, (start_x, start_y), font, font_scale, text_color, font_thickness)
                start_y += line_height + padding
                for answer, value in zip(answer_data['ans'], answer_data['val']):
                    text = f"{answer}: {value:.2f}"
                    cv2.putText(img_with_space_for_text, text, (start_x, start_y), font, font_scale, text_color, font_thickness)
                    start_y += line_height + padding
        else:
            # When a single dictionary is provided
            answer_data = answers_with_values
            for answer, value in zip(answer_data['ans'], answer_data['val']):
                text = f"{answer}: {value:.2f}"
                cv2.putText(img_with_space_for_text, text, (start_x, start_y), font, font_scale, text_color, font_thickness)
                start_y += line_height + padding
        return img_with_space_for_text