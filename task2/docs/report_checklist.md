# Report Checklist

Include the following in the homework report:

1. Dataset preparation: `xiaoma26/calvin-lerobot`, splitB for basic training, splitA/B/C for joint training, splitD for zero-shot evaluation.
2. ACT policy: ResNet-18 visual backbone, transformer encoder/decoder, action chunk size 100.
3. Logging: Weights & Biases training curves for loss and checkpoints.
4. Results: offline normalized action L1 on splitD for the basic policy and joint policy.
5. Discussion: whether A/B/C joint training improves cross-environment generalization to D.
