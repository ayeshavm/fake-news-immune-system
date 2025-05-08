# 🧠 Fake News Detection using Artificial Immune System (AIS)

This project applies the **Negative Selection Algorithm** from Artificial Immune Systems (AIS) to detect fake news articles using semantic embeddings — with **no labeled fake news required for training**.



## 🔍 Problem

Fake news is a major challenge for platforms and public trust. Traditional models rely heavily on labeled datasets. But in real-world scenarios, fake content evolves rapidly — often without labels.



## 💡 Solution

Inspired by the biological immune system, this project uses the **Negative Selection Algorithm** to:

- Generate "detectors" from known **real news** articles
- Flag incoming news as **fake (non-self)** if they activate a detector
- Use **cosine distance** over semantic embeddings to measure anomaly



## 🧪 Dataset

- **FakeNewsNet** (real + fake articles)
📚 [References](https://github.com/KaiDMML/FakeNewsNet)

- Shu, K., Mahudeswaran, D., Wang, S., Lee, D., & Liu, H. (2018). **FakeNewsNet: A Data Repository with News Content, Social Context and Dynamic Information for Studying Fake News on Social Media.** *arXiv preprint arXiv:1809.01286*. [arXiv link](https://arxiv.org/abs/1809.01286)

- Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). **Fake News Detection on Social Media: A Data Mining Perspective.** *ACM SIGKDD Explorations Newsletter*, 19(1), 22–36. [DOI](https://doi.org/10.1145/3137597.3137600)

- Shu, K., Wang, S., & Liu, H. (2017). **Exploiting Tri-Relationship for Fake News Detection.** *arXiv preprint arXiv:1712.07709*. [arXiv link](https://arxiv.org/abs/1712.07709)

✅ Includes
- Preprocessed and embedded using **spaCy GloVe** (300-d)
- Only **real articles** used for training detectors



## ⚙️ Approach

- Extract 300–500 detectors by adding controlled noise to real articles
- Use cosine distance to classify test articles as fake/real
- Sweep thresholds (0.5–0.6) to tune tradeoff between precision and recall



## 📊 Results

| Threshold | Fake Recall | Fake Precision | F1 (Fake) | Real Recall |
|-----------|-------------|----------------|-----------|--------------|
| 0.56      | 0.35        | 0.32           | 0.33      | 0.47         |
| 0.6       | 0.65        | 0.40           | 0.49      | 0.32         |

- Performance improved significantly over TF-IDF detectors
- Embeddings allowed for semantic anomaly detection
- Tradeoffs visible and tunable via threshold sweep



## 📈 Visuals

_(include your threshold vs. recall/precision/F1 plot here)_



## 🧬 Why AIS?

Unlike supervised models, this system:
- Requires **no labeled fake data**
- Can generalize to new topics/styles
- Is interpretable and tunable
- Inspired by real-world immune response design



## 🧩 Bonus (Optional, If Time Allows)

- [ ] Apply to **LIAR dataset**
- [ ] Compare with existing **MC Hybrid model** results
- [ ] Evaluate generalization across datasets



## 📂 Project Structure
```
/notebooks
- ais_detector_pipeline.ipynb
/plots
- threshold_vs_metrics.png
/data
- fakenewsnet.csv
README.md
```



