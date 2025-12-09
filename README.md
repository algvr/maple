# 🍁 MAPLE: Encoding Dexterous Robotic <ins>Ma</ins>nipulation <ins>P</ins>riors <ins>L</ins>earned From <ins>E</ins>gocentric Videos

## [[Paper]](https://arxiv.org/pdf/2504.06084)  [[Website]](https://algvr.com/maple/)

<img src="https://algvr.com/maple/static/images/maple_teaser.png" alt="Method Figure" style="width: 90%; margin-bottom: 8px;"/>

We present MAPLE, a framework that infuses dexterous manipulation priors from egocentric videos into vision encoders, making their features well-suited for downstream dexterous robotic manipulation tasks. 

# Using MAPLE

Download the MAPLE checkpoints using `download_checkpoints.sh`:

```bash
chmod +x download_checkpoints.sh
./download_checkpoints.sh
```


Use MAPLE as follows to extract features from images:

```python
from maple import MAPLE
from PIL import Image
import torch

if __name__ == "__main__":
    with torch.no_grad():
        checkpoint_path = "checkpoints/maple.pt"
        encoder = MAPLE(checkpoint_path)
        image = Image.open("example.jpg")
        features = encoder(image)
```


Stay tuned, more code coming soon!

# Citation

If you found our work useful, please cite it:

```
@article{gavryushin2025maple,
  title={{MAPLE: Encoding Dexterous Robotic Manipulation Priors Learned From Egocentric Videos}}, 
  author={Gavryushin, Alexey and Wang, Xi and Malate, Robert J. S. and Yang, Chenyu and Liconti, Davide and Zurbr{\"u}gg, Ren{\'e} and Katzschmann, Robert K. and Pollefeys, Marc},
  eprint={2504.06084},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  journal={arXiv preprint arXiv:2504.06084},
  year={2025},
  url={https://arxiv.org/abs/2504.06084}
}
```
