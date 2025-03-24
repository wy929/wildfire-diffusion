# Wildfire diffusion

## Environment

``` bash
conda env create -f environment.yml
```

## How to use

### Generate dataset

write or load a config file in `./config/data`

```bash
python3 scripts/dataset.py
```

### Train model

write or load a config file in `./config/train`

```
python3 scripts/train_ddim_unet.py
```



<img src="./assets/uml.svg" alt="uml" style="zoom:67%;" />

<p align="center"><img src="your-image-url.jpg" alt="" width="60%"></p><p align="center"><b>Figure 1:</b> (a) Ensemble sampling process; (b) generate probabilistic prediction undergoes
multiple diffusion inference processes.</p>
<img src="./assets/combined.gif" alt="combined" style="zoom: 33%;" />

<p align="center"><img src="your-image-url.jpg" alt="" width="60%"></p><p align="center"><b>Figure 2:</b> Result visualisation.</p>
