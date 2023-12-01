import torch

# データ設定，ディレクトリパス
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_EPOCH=5
TOCOMPARE_EPOCH=5 # must be same number as BASE_EPOCH
FLASHCARD_SAVE_IMG_PATH=f"flashcard/saved_img/epoch_{BASE_EPOCH}-{TOCOMPARE_EPOCH}" # flashcard app output path
FLASHCARD_CACHE_PATH=f"flashcard/cache/epoch_{BASE_EPOCH}-{TOCOMPARE_EPOCH}"
QA_DIR_BASE = f"../data/qa/base/vqa_output_special_qa_epoch{BASE_EPOCH}/epoch_{BASE_EPOCH}" # qa output from Lxmert, dir
QA_DIR_TOCOMPARE = f"../data/qa/tocompare/vqa_output_timo_qa_epoch{TOCOMPARE_EPOCH}/epoch_{TOCOMPARE_EPOCH}" # qa output from Mutan, dir
ORIG_IMG_DIR="../data/train2014"
GEND_IMG_DIR_BASE=f"../data/genim/base/vqa_output_special_genim_epoch{BASE_EPOCH}/epoch_{BASE_EPOCH}" # gen_im Lxmert dir
GEND_IMG_DIR_TOCOMPARE=f"../data/genim/tocompare/vqa_output_timo_genim_epoch{TOCOMPARE_EPOCH}/epoch_{TOCOMPARE_EPOCH}" # gen_im Mutan dir

# 動作テスト用
# GEND_IMG_DIR_BASE=f"../data/genim/base/vqa_output_special_genim_epoch{BASE_EPOCH}/test" # gen_im Lxmert dir
# GEND_IMG_DIR_TOCOMPARE=f"../data/genim/tocompare/vqa_output_timo_genim_epoch{TOCOMPARE_EPOCH}/test" # gen_im Mutan dir

# loss分析の設定
PERCEPTUAL_LOSS_BATCH_SIZE=128
L1_NORM_BATCH_SIZE=4096
NUM_OF_WORKERS=2
DATA_SHUFFLE=False
METRICS_TYPE = ['perceptual_loss', 'l1_norm']  # 必要に応じて 'perceptual_loss' と 'l1_norm' のいずれかまたは両方を含める
CALCULATED_LOSS_RESULT_PATH = f"loss_analysis/results/calculated_loss_result_{BASE_EPOCH}-{TOCOMPARE_EPOCH}.json"
ANALYSIS_RESULT_PATH = f"loss_analysis/results/analysis_result_{BASE_EPOCH}-{TOCOMPARE_EPOCH}.json"

# キャッシュの設定
ENABLE_CACHE = True  # キャッシュを有効にするか
REGENERATE_CACHE = True  # キャッシュを再生成するか

# flashcard app
ORIG_IMG_LABEL_NAME="Original"
BASE_IMG_LABEL_NAME="w/Lxmert"
TOCOMPARE_IMG_LABEL_NAME="w/Mutan"
KEY_INSTRUCTIONS="Key Instructions | n: Next Image | b: Previous Image | d: Auto Detect | s: Save Image | x: Continuous Auto-Save | Space: Pause/Resume  | q: Quit"
METRICS_LABEL_NAME='Perceptual Dist'
