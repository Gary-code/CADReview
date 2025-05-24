# CADReview
[ACL 2025 Main] The official repository of our paper: CADReview: Automatically Reviewing CAD Programs with Error Detection and Correction

# Training and Inference

1. Our training and inference are conducted using the [ms-swift](https://github.com/modelscope/ms-swift) framework. Environment configuration: `ms-swift >= 3.3`, `vllm >= 0.7.3`.
2. The alignment training for GCR and SGO can be found in:
   `./training_and_inference/alignment_gcr` and `./training_and_inference/alignment_sgo`.
3. Training for $\phi_1$ and $\phi_2$ can be found in:
   `./training_and_inference/feedback_gen` and `./training_and_inference/code_editor`.
4. The inference script can be found at:
   `./training_and_inference/inference.py`.

# Evaluation

Run `./evaluate/eval.sh` to perform evaluation.

