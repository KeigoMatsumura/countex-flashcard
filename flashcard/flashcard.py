import os
import cv2
import config
from flashcard.data_processor import DataProcessor
from flashcard.window_generator import WindowGenerator

class FlashCard:
    def __init__(self, analized_data):
        self.analized_data = analized_data
        self.data_processor = DataProcessor(
            config.ORIG_IMG_DIR,
            config.GEND_IMG_DIR_BASE,
            config.GEND_IMG_DIR_TOCOMPARE,
            config.QA_DIR_BASE,
            config.QA_DIR_TOCOMPARE,
            config.FLASHCARD_SAVE_IMG_PATH,
        )
        self.window_generator = WindowGenerator()
        self.size = (256,256)
        self.delay = 10
        self.filter_rate = 0.8
        self.index = 0
        self.paused = False # プログラムが一時停止するフラグ
        self.auto_next = False # 自動で次の画像に進むためのフラグ
        self.auto_detect = False # 自動で回答１と回答２の出力に、変更があったかを検出するためのフラグ
        self.input_mode = False # 入力モード切り替え
        self.found = False        

    def show(self):
        while True:  # 無限ループ
            if self.index >= len(self.data_processor.img_data['base_img_paths']):  # インデックスがリストを超えた場合
                break  # ループを抜ける

            images, qas = self.data_processor.get_image_and_qa_data(self.index)
            image_id = images['img_id']

            if self.auto_next and not self.paused:
                if self.auto_detect and self.data_processor.is_filter_passed(image_id, self.analized_data):
                    self.auto_next = False
                else:
                    if self.index < len(self.data_processor.img_data['base_img_paths']) - 1:
                        self.index += 1  # 次の画像へ
                    else:
                        self.auto_next = False

            if self.data_processor.is_filter_passed(image_id, self.analized_data):
                # import pdb; pdb.set_trace()
                images['base_img'] = self.data_processor.add_red_border(images['base_img']) # 赤枠を追加
                images['tocompare_img'] = self.data_processor.add_red_border(images['tocompare_img']) # 赤枠を追加

            window_img = self.window_generator.generate(images, qas, self.analized_data)

            cv2.imshow('Image Flashcard', window_img)
            cv2.setWindowTitle('Image Flashcard', f'Image ID: {image_id}')

            key = cv2.waitKey(self.delay if not self.paused else 0)  # 一時停止している場合は無限に待機、それ以外は指定されたミリ秒だけ待機
            key = key & 0xFF
            self.found = False
            # ID入力モードの開始
            if key == ord('/'):
                self.input_mode = True
                input_id = ""
                print("Enter Image ID:")

            # ID入力モード中に数字が入力された場合
            elif self.input_mode and key in [ord(str(i)) for i in range(10)]:
                input_id += chr(key)

            # ID入力モード終了（Enterキー）
            elif self.input_mode and key == ord('\r'):
                self.input_mode = False
                # IDに基づいて画像インデックスを検索
                for idx, path in enumerate(self.data_processor.img_data['base_img_paths']):
                    if os.path.splitext(os.path.basename(path))[0].split('_')[-1] == input_id:
                        self.index = idx
                        self.found = True
                        break
                if self.found:
                    print(f"Jumping to Image ID: {input_id}, Index: {self.index}")
                else:
                    print(f"Image ID: {input_id} not found.")

            elif key == ord('n'):
                if self.index < len(self.data_processor.img_data['base_img_paths']) - 1:
                    self.index += 1  # 次の画像へ

            elif key == ord('b'):
                if self.index > 0:
                    self.index -= 1  # 一つ前の画像へ

            elif key == ord('s'):  # 's' key
                save_image_file_name = f"ImgID_{image_id}.jpg"
                save_image_file_name = os.path.join(config.FLASHCARD_SAVE_IMG_PATH, save_image_file_name)
                cv2.imwrite(save_image_file_name, window_img)
                print(f"Image saved as {save_image_file_name}")

            elif key == ord('d'):  # Space bar
                self.paused = not self.paused  # 一時停止/再開
                if not self.paused:
                    self.auto_next = True  # 再開時に自動進行を始める
                    self.auto_detect = True # 再開時に自動検出を始める

            elif key == ord(' '): 
                self.paused = not self.paused  # 一時停止/再開
                if not self.paused:
                    self.auto_next = True  # 再開時に自動進行を始める
                    self.auto_detect = False
                
            elif key == ord('q') or key == 27:  # 'q' key or Escape key
                break

            # print("index: " + str(self.index))  # デバッグ用の出力、必要なければコメントアウトする

        cv2.destroyAllWindows()

        return self.input_mode