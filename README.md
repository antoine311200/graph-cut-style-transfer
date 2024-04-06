# graph-cut-style-transfer
Reimplementation of the paper "Multimodal Style Transfer via Graph Cuts" from Yulun Zhang, Chen Fang, Yilin Wang, Zhaowen Wang, Zhe Lin, Yun Fu, Jimei Yang

## Usage
```bash
python -m src.test --n_clusters=3 --alpha=1.0 --lambd=0.1 --algo="ae" --distance="cosine" --content="./data/content/004.jpg" --style="./data/style/10.jpg"
```
