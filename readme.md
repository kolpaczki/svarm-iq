## 📄 Supplement Material for SVARM-IQ Submission.
This repository holds the supplement material for the contribution 
_SVARM-IQ: Efficient Approximation of Any-order Shapley-Interactions through Stratification_.

### 🚀 Quickstart
For a quick introduction, we refer to the `main.ipynb` notebook:
Install the dependencies via `pip install -r requirements.txt` and run the notebook.

#### Setup
```python
from games import LookUpGame
from approximators import InterSVARM as SVARMIQ

game = LookUpGame(
    data_folder="vision_transformer",   # what explanation task / model to use
    n=16                                # number of image patches for the vision transformer
)

game_name = game.game_name
game_fun = game.set_call
n_players = game.n
player_set = set(range(n_players))

interaction_order = 2
```

#### SVARM-IQ to approximate the Shapley Interaction Index

```python
svarmiq_sii = SVARMIQ(
    interaction_type="SII", 
    N=player_set,
    order=interaction_order, 
    top_order=False
)

sii_scores = svarmiq_sii.approximate_with_budget(game=game_fun, budget=budget)
```
#### SVARM-IQ to approximate the Shapley Taylor Index

```python
svarmiq_sti = SVARMIQ(
    interaction_type="STI", 
    N=player_set, 
    order=interaction_order,  
    top_order=True
)

sti_scores = svarmiq_sti.approximate_with_budget(game=game_fun, budget=budget)
```
#### SVARM-IQ to approximate the Shapley Faith Index

```python
svarmiq_fsi = SVARMIQ(
    interaction_type="FSI",
    N=player_set,
    order=interaction_order, 
    top_order=True
)

fsi_scores = svarmiq_fsi.approximate_with_budget(game=game_fun, budget=budget)
```

### ✅ Validate Experiments

To run and validate the same experiments as in the paper the lookup data may need to be first pre-computed.
For this we refer to `precompute_lm.py`,  `precompute_cnn.py`, and `precompute_vit.py`.
We provide some example images and sentences in the `games/data` folder.

The experiments can be run via the files with the prefix `exp_` in the root folder.
