# ROVER Recipe

This recipe implements ROVER (Random Policy Valuation for Diverse Reasoning) as described in
arXiv:2509.24981. The core idea is to regress relative Q-values using the uniform-policy
mean backup, where Q is parameterized by the LLM log-prob change against a frozen policy.

## Defaults

Base defaults live in `../verl/recipe/rover/config/rover_trainer.yaml`:
- `actor.rover_rho: 1.0` (temperature scaling for relative Q)
- `actor.rover_q_next_coef: 1.0` (scales the mean-Q next-state term)
- `actor.use_kl_loss: false`
- `actor.entropy_coeff: 0.0`
- `rollout.n: 4`

Example Llama config in this repo is `train/configs/train_llama_rover.yaml`, which overrides
model, dataset paths, and uses `rollout.n: 16`.

## Key knobs

- `actor.rover_rho`: scales `Q = rho * (log pi_theta - log pi_old)`.
- `actor.rover_q_next_coef`: multiplies the mean-Q bootstrap; keep in `[0.2, 1.0]`.
- `rollout.n`: number of responses per prompt; must be >1 for centered rewards.
- `actor.ppo_mini_batch_size`, `actor.ppo_epochs`: control how many updates per batch.
- `actor.use_kl_loss`: keep `false` for pure ROVER.
- `model.use_fused_kernels`: must be `false` because ROVER needs full logits.

## Off-policy by one step

ROVER computes `pi_old` (old log-probs) once per batch, then runs multiple
optimizer steps on that batch. Any step after the first is off-policy relative to `pi_old`.

To make it "one step off-policy", keep `actor.ppo_epochs = 1` and set
`actor.ppo_mini_batch_size` so the batch splits into **two** minibatches:

- Example: `train_batch_size = 64`, `ppo_mini_batch_size = 32`
- First minibatch uses `pi_old` (on-policy), second minibatch is one update off-policy.

More off-policy steps: decrease `ppo_mini_batch_size` (more minibatches) or increase
`ppo_epochs`.

## Run

From a repo that includes the ROVER config in the Hydra search path:
```
python -m recipe.rover.main_rover --config-name rover_trainer
```

For Llama 3.1 8B Instruct, see `experiments/train_llama_rover.sh` in this repo.
