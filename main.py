
from torch.utils.data import DataLoader
import config
from loss_analysis.analysis import Analysis
from loss_analysis.dataset import ImagePairDataset
from flashcard.flashcard import FlashCard

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

    flashcard = FlashCard(analyzed_output)
    flashcard.show()
    
if __name__ == "__main__":
    main()