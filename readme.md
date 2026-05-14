# InstructPix2Pix for Robotic Frame Prediction

This project fine-tunes the [InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix) model to predict future frames in robotic manipulation tasks, given a current image and a textual instruction. The fine-tuned model generates what the robot is expected to see 50 frames later.

## Project Structure

```
instructpix2pix/
├── dataset/                      // original data
├── data/                         // processed data
│   ├── train/sceneX/
│   ├── validation/sceneX/
│   └── test/
├── robot_arm_model_sceneX/       // fine-tuned models
├── sample_figures/
├── results/                      // evaluation results
├── scripts/
│   ├── finetune.py
│   ├── eval.py
│   ├── sample_figures.py
│   ├── make_train_data.py
│   ├── make_test_data.py
│   ├── process_robot_dataset.py
│   └── reprocess_dataset.py
├── requirements.txt
└── README.md
```

## Dependencies

Install required packages with:

```bash
pip install -r requirements.txt
```

Tested versions:

- Python 3.12
- PyTorch 2.3.0 with CUDA 12.2
- `diffusers==0.33.1`
- `transformers==4.51.3`
- `accelerate==1.6.0`
- `scikit-image`, `Pillow`, `matplotlib`, `tqdm`

## Fine-Tuning

To fine-tune the model for a specific task (e.g., `scene4`), run:

```bash
export SCENE_ID=4
python scripts/finetune.py
```

Outputs (checkpoints, logs) will be saved in `robot_arm_model_scene4/`.

## Evaluation

Evaluate selected models (e.g., model 0–3) on the test set:

```bash
python scripts/eval.py --models 0 1 2 3
```

Results (SSIM/PSNR) and generated images are saved to `results/`.

## Sample Comparison

To visualize predictions from multiple models:

```bash
python scripts/sample_figures.py --models 0 3 4 5 --num_samples 10
```

Images will be saved in `sample_figures/` with side-by-side comparisons.


## Contact

- Tian Zeyu: [183441801@qq.com](mailto:183441801@qq.com)


## Citation

Please cite the original InstructPix2Pix model and this project if you find it useful.
