# Config Files

## best_config.json

`best_config.json` stores the hyperparameters used by the B3 and B3+DRKG models.

It is loaded by:

- `run_all.ps1`
- `run_pancancer_all.ps1`
- `src/run_attentivefp_multiseed_cv.py`

Main fields:

- `lr`: learning rate
- `weight_decay`: L2 regularization
- `dropout`: dropout rate
- `gnn_hidden`: AttentiveFP hidden dimension
- `rank`: LMF fusion rank
- `fusion_dim`: fusion hidden dimension
- `gnn_layers`: number of GNN layers
- `gnn_timesteps`: AttentiveFP message-passing timesteps
- `alpha`: default multitask loss weight
- `patience`: early stopping patience

For the final reported experiments, the command line uses `--alphas 0.8`, so the reported runs use `alpha = 0.8` even though this JSON records the Optuna-selected default alpha.
