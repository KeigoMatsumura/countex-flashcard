
from torch.utils.data import DataLoader
import config
from loss_analysis.analysis import Analysis
from loss_analysis.dataset import ImagePairDataset
from flashcard.flashcard import FlashCard
import os, json

# JSONデータの読み込み関数
def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def find_qa_file_for_image(qa_directory, img_id):
    qa_filepath = os.path.join(qa_directory, f"qa_{img_id}.json")
    if os.path.exists(qa_filepath):
        return load_json(qa_filepath)
    else:
        return None

# メインの処理を関数化
def main():
    print("Starting main processing...")  # 開始メッセージ
    dataset = ImagePairDataset(config.ORIG_IMG_DIR, config.GEND_IMG_DIR_BASE, config.GEND_IMG_DIR_TOCOMPARE)
    perceptual_loss_loader = DataLoader(dataset, batch_size=config.PERCEPTUAL_LOSS_BATCH_SIZE, shuffle=config.DATA_SHUFFLE)
    l1_norm_loader = DataLoader(dataset, batch_size=config.L1_NORM_BATCH_SIZE, shuffle=config.DATA_SHUFFLE)

    analyzed_output = {}
    analysis = Analysis(metrics_types=config.METRICS_TYPE, device=config.DEVICE)

    # Perceptual Lossの計算
    if 'perceptual_loss' in config.METRICS_TYPE:
        # Perceptual Lossの計算と処理
        analysis.check_and_process_loss(perceptual_loss_loader, 'perceptual_loss')

    # l1 normの計算
    if 'l1_norm' in config.METRICS_TYPE:
        analysis.check_and_process_loss(l1_norm_loader, 'l1_norm')
    
    analyzed_output = analysis.loss_data

    # Iterate through all image IDs and process QA data
    for img_id in analysis.loss_data.keys():
        # Process QA Data for BASE
        qa_data_base = find_qa_file_for_image(config.QA_DIR_BASE, img_id)
        if qa_data_base:
            analysis.process_qa_data(qa_data_base, analysis.loss_data[img_id])

        # Process QA Data for TOCOMPARE
        qa_data_tocompare = find_qa_file_for_image(config.QA_DIR_TOCOMPARE, img_id)
        if qa_data_tocompare:
            analysis.process_qa_data(qa_data_tocompare, analysis.loss_data[img_id])

    # Calculate and Save Results
    analysis.analyze_and_save_results(config.ANALYSIS_RESULT_PATH)

    flashcard = FlashCard(analyzed_output)
    flashcard.show()
    
if __name__ == "__main__":
    main()