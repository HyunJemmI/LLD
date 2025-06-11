# Light Noise Robust Lane detection model based on Contrastive Learning

Lane Detection with SimCLR‑pretrained LaneATT

## 1. 설치
```bash
pip install torch torchvision lightly pillow numpy
```

## 2. 실행 순서
```bash
# (1) SimCLR 사전학습
python training/train_contrastive.py

# (2) Baseline 학습
python training/train_baseline.py

# (3) Fine‑tune (SimCLR encoder freeze)
python training/train_finetune.py

# (4) 평가 (accuracy / FPS)
python experiments/eval_lane_detection.py
```