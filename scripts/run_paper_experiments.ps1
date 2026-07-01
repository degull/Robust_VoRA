$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

$RestormerCkpt = "checkpoints/restormer/deraining.pth"
$PatchSize = 128
$BatchSize = 1
$Epochs = 20
$Rank = 8
$CodeDim = 64
$LambdaFeat = 1.0
$EvalSeverity = 0.75
$LogInterval = 100

python scripts/train_baseline.py `
  --model adaptive_lora `
  --restormer-checkpoint $RestormerCkpt `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --patch-size $PatchSize `
  --vora-rank $Rank `
  --vora-code-dim $CodeDim `
  --lambda-feat $LambdaFeat `
  --valid-severity $EvalSeverity `
  --log-interval $LogInterval `
  --output-dir outputs/paper/adaptive_lora_lam1

python scripts/train_baseline.py `
  --model adaptive_vora `
  --restormer-checkpoint $RestormerCkpt `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --patch-size $PatchSize `
  --vora-rank $Rank `
  --vora-code-dim $CodeDim `
  --lambda-feat $LambdaFeat `
  --valid-severity $EvalSeverity `
  --log-interval $LogInterval `
  --output-dir outputs/paper/adaptive_vora_lam1

python scripts/compare_lora_vora_features.py `
  --restormer-checkpoint $RestormerCkpt `
  --lora-checkpoint outputs/paper/adaptive_lora_lam1/best.pt `
  --vora-checkpoint outputs/paper/adaptive_vora_lam1/best.pt `
  --batch-size 1 `
  --patch-size $PatchSize `
  --rank $Rank `
  --code-dim $CodeDim `
  --eval-severity $EvalSeverity `
  --visual-perturbation snow `
  --visual-layer model.encoder_level2 `
  --visual-image data/Flickr2K/000263.png `
  --feature-upsample 8 `
  --output-dir outputs/paper/compare_lora_vora

python scripts/analyze_feature_shift.py `
  --restormer-checkpoint $RestormerCkpt `
  --vora-checkpoint outputs/paper/adaptive_vora_lam1/best.pt `
  --batch-size 1 `
  --patch-size $PatchSize `
  --vora-rank $Rank `
  --vora-code-dim $CodeDim `
  --eval-severity $EvalSeverity `
  --visual-image data/Flickr2K/000263.png `
  --output-dir outputs/paper/feature_shift_vora
