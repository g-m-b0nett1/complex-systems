# Data

The network study uses two public data sets from the Stanford Network Analysis
Project (SNAP). They are not stored in the repository; download them with

```bash
python scripts/fetch_data.py
```

| File | Network | Nodes | Edges |
|---|---|---|---|
| `facebook_combined.txt.gz` | [ego-Facebook](https://snap.stanford.edu/data/ego-Facebook.html) | 4,039 | 88,234 |
| `roadNet-PA.txt.gz` | [Pennsylvania road network](https://snap.stanford.edu/data/roadNet-PA.html) | 1,088,092 | 1,541,898 |

Plain `.txt` and gzipped `.txt.gz` files are both accepted.
