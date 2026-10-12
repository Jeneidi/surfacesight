# SurfaceSight

Transfer-learned visual inspection for manufactured bottle surfaces, with an ONNX int8 export that keeps a 0.96 held-out defect F1 and runs at a measured 1.628792 ms median on CPU.

Live demo: https://surfacesight.mjeneidi.com

## The problem

Plant quality teams need to catch scratches, dents and contamination before products ship. Manual inspection is tiring, while a model that is accurate but too slow is not useful near a production line.

## The result

On 59 untouched held-out bottle images, the fine-tuned ResNet18 reached defect F1 0.96 versus 0.8695652173913043 for a frozen ImageNet feature baseline. The int8 ONNX model also reached 0.96 F1, reduced the model from 44,788,648 bytes to 11,231,354 bytes and reduced median CPU latency from 8.757834 ms to 1.628792 ms.

## How it works

1. Download the bottle category from a public file-level MVTec AD mirror, keeping raw images outside the static site.
2. Split 292 labeled images into a seeded 80/20 train and holdout split with no filename overlap.
3. Compare a frozen ImageNet ResNet18 feature baseline with a two-epoch CPU fine-tune of layer 4 and the classifier head.
4. Export the fine-tuned model to ONNX, calibrate a QDQ int8 model and benchmark both paths on CPU.
5. Publish recorded predictions, confidence, the held-out mistake and the export checks in a static page.

## Data and license

The benchmark is the MVTec AD bottle category, originally published by MVTec Software GmbH. The official dataset page is https://www.mvtec.com/research-teaching/datasets/mvtec-ad. This build uses the public bottle files at https://huggingface.co/datasets/foersben/mvtec-ad because the official page requires a download form and the brief prohibits contacting anyone. The dataset is licensed under CC BY-NC-SA 4.0. It is not licensed for commercial use.

## Evaluation method

The headline metric is binary defect F1 on the untouched holdout. The baseline is logistic regression over frozen ImageNet ResNet18 features. The fine-tuned model updates ResNet18 layer 4 and its classifier head on the training split. The int8 comparison uses the same holdout images. The fixed sample export check recorded a maximum absolute logit difference of 0.9363489151000977 against a 1.0 tolerance, and the PyTorch and ONNX predictions agreed.

## How to run

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
./.venv/bin/python scripts/download_data.py
./.venv/bin/python scripts/train.py
./.venv/bin/pytest -q
cd site
python3 -m http.server 8000 --bind 127.0.0.1
```

## Limitations

MVTec AD is a controlled benchmark, not a production line. The public mirror is used to keep the build small and avoid the official form, so the raw source should be checked before any commercial or redistributed use. The run covers one category, uses two training epochs and reports image-level classification rather than pixel-level localization. The site is a recorded static demo and does not claim production readiness.

## Credits

Dataset: Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger, MVTec AD. Model: ResNet18 ImageNet weights from torchvision. Runtime: PyTorch and ONNX Runtime. Site: Project Site Kit, plain HTML, CSS and JavaScript.

