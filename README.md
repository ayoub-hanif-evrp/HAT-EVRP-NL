# HAT-EVRP-NL

Compact research implementation for the Montoya et al. E-VRP with nonlinear
charging. It includes an independently checked `frvcpy` charging oracle,
improved classical H0, H1-H4 neural ablations, and a secondary learned
GO/CHARGE policy.

## Reproduce

```bash
pip install -r requirements.txt
python evaluate.py --mode sanity --data dataset
python evaluate.py --mode baseline --size 10 --limit 20 --restarts 10 --local-passes 3 --local-budget 4000 --seed 42

python train.py --model H1 --size 10 --steps 200 --rl-steps 20 --train-instances 30 --val-instances 5 --seed 42
python train.py --model H2 --size 10 --steps 200 --rl-steps 20 --train-instances 30 --val-instances 5 --seed 42
python train.py --model H3 --size 10 --steps 200 --rl-steps 20 --train-instances 30 --val-instances 5 --seed 42
python train.py --model H4 --size 10 --steps 200 --rl-steps 20 --train-instances 30 --val-instances 5 --seed 42
python evaluate.py --mode compare --size 10 --limit 20 --samples 10 --seed 42

python train.py --model CHARGE --size 10 --steps 2000 --train-instances 30 --seed 42
python evaluate.py --mode direct --size 10 --limit 20 --samples 10 --seed 42
python evaluate.py --mode plots --seed 42
```

Executed numerical results, limitations, exact experimental settings, figures,
and a fully feasible concrete route are under `results/`.
