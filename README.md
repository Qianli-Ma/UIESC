# UIESC — Thesis Baseline

Standalone reimplementation of UIESC (Chen et al., 2023) retrained on UIEB for use as
a Transformer-based baseline in the thesis comparison against TransFuse-GAN.

> Chen, R., Cai, Z., & Yuan, J. (2023). UIESC: An underwater image enhancement
> framework via self-attention and contrastive learning. *IEEE Transactions on
> Industrial Informatics*, *19*(9), 9844–9853.
> https://doi.org/10.1109/TII.2022.3230239

No public implementation was released by the authors. This reimplementation is based
on the architecture and training details described in the paper. All design decisions not
specified in the paper are documented as assumptions in the table below and exposed as
configurable parameters in `config.yaml`.

---

## Thesis Context

UIESC is included as a self-attention- and contrastive-learning-based baseline in the
thesis comparison. It is evaluated on the UIEB test split alongside FUnIE-GAN,
UW-CycleGAN, and UDAformer. Metrics reported are PSNR, SSIM, UCIQE, and UIQM,
computed by the shared `evaluation/` module.

---

## Architecture

| Component | Setting | Source |
|---|---|---|
| Input / output | `(B, 3, H, W)`, values `[0, 1]`; output same shape and range | Paper pipeline |
| Stage 1 | LSAN autoencoder: downsampling, three LSA placements, upsampling, AFF, residual output | Paper §III-A |
| LSA spatial branch | Criss-cross row and column self-attention | Paper §III-A1 |
| LSA channel branch | Channel self-attention with learnable scale `beta` | Paper Eq. 5–6 |
| AFF fusion | `sigmoid(λ_i) · f_down + sigmoid(1 − λ_i) · f_up` | Paper Eq. 7 |
| Residual learning | Predict residual, add to input, clamp to `[0, 1]` | Paper §III-A |
| Stage 2 | Smoothed histogram equalization (SHE), applied after LSAN | Paper §III-B |
| SHE bins | 256 | 8-bit histogram assumption |
| Base channels | 32 | **Assumed** — compact default; not stated in paper |
| LSA repeats | `[1, 1, 2]` | **Assumed** — three LSA stages plus bottleneck refinement |
| Loss weights | `λ_l1 = 1`, `λ_ms-ssim = 0.01`, `λ_cl = 0.01` | Paper Eq. 15 |
| Contrastive feature network | VGG-19 | Paper Eq. 14 |
| Contrastive layers | VGG feature indices `[1, 3, 5, 9, 13]` | **Assumed** — common setting from cited compact dehazing loss |
| Contrastive weights ω_i | `[1/32, 1/16, 1/8, 1/4, 1]` | **Assumed** — same cited setup |
| Optimizer | Adam | Paper §IV-A |
| Learning rate | `2e-4` | Paper §IV-A |
| Training epochs | 200 (UIEB thesis baseline); paper reports 2000 on EUVP + UIEB | Config / paper §IV-A |

**Ambiguities.** The paper specifies the LSA equations, AFF equation, SHE procedure,
losses, optimizer, and learning rate. Exact channel widths and per-block kernel
schedules are not available in extractable text from the PDF and are documented above
as assumptions. The contrastive loss layer indices and weights are not tabulated in the
paper; this implementation uses the common setting from the compact dehazing
contrastive loss cited by the paper.

---

## Environment

```bash
mamba create -n uiesc python=3.11.15 -y
mamba install -n uiesc pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
mamba run -n uiesc pip install tensorboard timm scikit-image Pillow tqdm pyyaml
```

Verified package versions:

| Package | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.11.0 |
| torchvision | 0.26.0 |
| tensorboard | 2.20.0 |
| timm | 1.0.27 |
| scikit-image | 0.26.0 |
| Pillow | 12.2.0 |
| tqdm | 4.68.2 |
| PyYAML | 6.0.3 |

CUDA was not available in the verification session, though the environment was created
with `pytorch-cuda=12.1`.

---

## Local Copies

The UIEB loader and MS-SSIM loss are vendored into `dataset.py` and `loss.py`.
This baseline does not import helper modules from TransFuse-GAN at runtime.

Expected layout:

```
UIESC/
    model.py
    train.py
    dataset.py
    loss.py
    config.yaml
    outputs/
        checkpoints/
```

---

## Datasets

**UIEB** (Li et al., 2020): 890 paired underwater images. Download from the
[project page](https://li-chongyi.github.io/proj_benchmark.html) and set
`uieb_input` and `uieb_target` in `config.yaml`. The default paths point to
`experiments/datasets/UIEB/raw-890`, `reference-890`, and the split files in
`experiments/datasets/UIEB`.

---

## Training

```bash
mamba run -n uiesc python experiments/models/baselines/UIESC/train.py \
    --config experiments/models/baselines/UIESC/config.yaml
```

The best validation-PSNR checkpoint is saved to:

```
experiments/models/baselines/UIESC/outputs/checkpoints/best_generator.pth
```

Validation runs `model(inputs)` so SHE is active during checkpoint selection.
Training losses are applied to `model.forward_lsan(inputs)` before SHE, matching
the paper's two-stage design.

To resume from a checkpoint, pass `--resume`.

---

## Evaluation

```bash
mamba run -n uiesc python eval.py \
    --checkpoint experiments/models/baselines/UIESC/outputs/checkpoints/best_generator.pth \
    --split test \
    --device cuda
```

Prints PSNR, SSIM, UCIQE, UIQM, parameter count, ms/image, and peak VRAM.

---

## Inference Stats

```bash
mamba run -n uiesc python experiments/models/baselines/UIESC/train.py \
    --config experiments/models/baselines/UIESC/config.yaml --inference-stats
```

Prints AMP status, parameter count, total test images, wall-clock time per image at
batch size 1, and peak VRAM.

---

## Smoke Test

```bash
cd experiments/models/baselines/UIESC
mamba run -n uiesc python model.py
```

Expected output:

```
output shape: (1, 3, 256, 256)
parameter count: X.XXX M
```

---

## Citation

```bibtex
  @article{chenUIESCUnderwaterImage2023,
  title = {{{UIESC}}: {{An Underwater Image Enhancement Framework}} via {{Self-Attention}} and {{Contrastive Learning}}},
  shorttitle = {{{UIESC}}},
  author = {Chen, Renzhang and Cai, Zhanchuan and Yuan, Jieyu},
  year = 2023,
  month = dec,
  journal = {IEEE Transactions on Industrial Informatics},
  volume = {19},
  number = {12},
  pages = {11701--11711},
  issn = {1941-0050},
  doi = {10.1109/TII.2023.3249794},
  urldate = {2026-01-02},
}
```
