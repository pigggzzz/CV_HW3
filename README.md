## 题目一：基于 2DGS 与 AIGC 的多源资产生成与真实场景融合


https://github.com/user-attachments/assets/979ce9c1-6711-43dc-a38b-54441dc675bd

漫游渲染视频

## 题目二：基于LeRobot 的ACT 策略跨环境泛化挑战

https://github.com/pigggzzz/CV_HW3/tree/master/ACT/asset/b/seqid_0024_start_1536_mean_l1_0.555

```bash
conda create -n ACT python=3.12
conda activate ACT
pip install -r requirements.txt
```

训练指令：
```bash
bash scripts/train_basic.sh \
  --cuda-id 0 \
  --data-dir /home/lama/task2/data \
  --output-dir /home/lama/task2/output \
  --wandb-mode online

bash scripts/train_joint.sh \
  --cuda-id 0 \
  --data-dir /home/lama/task2/data \
  --output-dir /home/lama/task2/output \
  --wandb-mode online
```

zero-shot指令
```bash
bash scripts/eval_zero_shot.sh \
  --cuda-id 0 \
  --data-dir /home/lama/task2/data \
  --output-dir /home/lama/task2/output \
  --wandb-mode online
```

可视化指令：
```bash
bash scripts/offline_replay_visualize.sh \
  --cuda-id 0 \
  --data-dir /home/lama/task2/data \
  --output-dir /home/lama/task2/output \
  --sequence-ids 20,24,27,30 \
  --steps-per-sequence 32 \
  --min-valid-horizon 32 \
  --wandb-mode offline
```
