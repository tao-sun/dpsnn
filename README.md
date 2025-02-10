# DPSNN: Spiking Neural Network for Low-Latency Streaming Speech Enhancement

If you use this code in an academic context, please cite our work:

```bibtex
@article{sun2024dpsnn,
  title={DPSNN: spiking neural network for low-latency streaming speech enhancement},
  author={Sun, Tao and Boht{\'e}, Sander},
  journal={Neuromorphic Computing and Engineering},
  volume={4},
  number={4},
  pages={044008},
  year={2024},
  publisher={IOP Publishing}
}
```

## Installation
Follow the steps in installation.txt.

## Training and Inference
```bash
cd egs/voicebank
# Training and testing
python -u vctk_trainer.py --config vctk.yaml -L 80 --stride 40 -N 256 -B 256 -H 256 --context_dur 0.01 --max_epochs 500 -X 1 --lr 1e-2
# Inference only
python -u vctk_trainer.py --config vctk.yaml -L 80 --stride 40 -N 256 -B 256 -H 256 --context_dur 0.01 --max_epochs 500 -X 1 --lr 1e-2 --test_ckpt_path ./epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt
```
The model file for this run is *epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt*. Note that the model itself may differ slightly across different versions of PyTorch.

## Demos
Demos are in the <em>audio_demos</em> folder.