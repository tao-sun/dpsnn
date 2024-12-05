# dpsnn

## Installation
Follow the steps in installation.txt.

## Train
```bash
cd egs/voicebank
python -u vctk_trainer.py --config vctk.yaml -L 80 --stride 40 -N 256 -B 256 -H 256 --context_dur 0.01 --max_epochs 500 -X 1 --lr 1e-2
```

## Demos
Demos are in the <em>audio_demos</em> folder.