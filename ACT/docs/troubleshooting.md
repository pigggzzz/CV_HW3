# Troubleshooting

## `action_feature is None`

The dataset does not have a standard LeRobot ACT output key named exactly `action`. Run:

```bash
python -m crossenv_act inspect --data-dir ./data --output-dir ./output --force-prepare
```

Then inspect `act_schema_report.json` under the prepared split roots.

## `KeyError: image`

The visual key exists in `meta/info.json` but is missing from `meta/stats.json`. The schema fixer adds ImageNet-style visual stats. Rebuild with:

```bash
python -m crossenv_act inspect --data-dir ./data --output-dir ./output --force-prepare
```

## `splitA_old_old...`

This package does not scan converter backup directories. It uses `calvin_act_work/raw` and `calvin_act_work/prepared`. Old folders from previous packages can be ignored, or removed manually if disk space matters.

## `unsupported operand type(s) for +: int and NoneType` in ACT forward

Do not use training forward for zero-shot inference. This package's eval code uses `predict_action_chunk()` and action L1 instead.
