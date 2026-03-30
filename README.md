# Meeting Transcript Efficiency Analyzer
**CAP5771 – Intro to Data Science | Spring 2026**
Rahul Reddy Asam · Ashruth Reddy Gangula · Rakesh Kumar Reddy Dodda

---

## What This Project Is About

We've all sat through meetings that felt like they went nowhere. The frustrating part is that it's hard to prove — it's just a feeling. This project tries to change that by actually looking at what gets said in a meeting and measuring whether it was useful.

We use the AMI Meeting Corpus, which contains transcripts of 171 real workplace meetings. For each one, we calculate three things:

- **Decision density** — how often did someone actually commit to something?
- **Redundancy score** — how much of the meeting was just re-covering old ground?
- **Action clarity** — when tasks were assigned, did anyone actually get named?

Those three numbers feed into a single efficiency score that we use to rank and cluster meetings. No surveys. No vibes. Just what was said.

---

## Repo Layout

```
CAP5771_Meeting-Transcript-Analyzer/
│
├── code/
│   ├── milestone1_analysis.ipynb        # data download, EDA, SQLite setup
│   ├── data_wrangling.ipynb             # cleaning, labeling, feature building
│   ├── data_modeling.ipynb              # 5 classifiers + meeting clustering
│   └── data_visualization_static.ipynb  # dashboard (static panels + widgets)
│
├── data/
│   ├── raw/                             # transcript .txt files — generated at runtime
│   ├── processed/
│   │   ├── utterance_level_data.csv     # from milestone1
│   │   ├── meeting_level_features.csv   # from milestone1
│   │   ├── analysis_ready.csv           # from data_wrangling
│   │   ├── meeting_metrics.csv          # from data_wrangling
│   │   ├── analysis_ready_with_preds.csv  # from data_modeling
│   │   └── meeting_metrics_clustered.csv  # from data_modeling
│   └── meeting_transcripts.db           # SQLite database
│
├── models/
│   ├── best_classifier.pkl
│   ├── label_encoder.pkl
│   ├── kmeans_clusters.pkl
│   └── cluster_scaler.pkl
│
├── outputs/
│   ├── model_comparison.png
│   ├── confusion_matrices_all.png
│   ├── feature_importance.png
│   ├── kmeans_selection.png
│   ├── meeting_clusters.png
│   ├── model_comparison_summary.csv
│   └── error_analysis.csv
│
├── diary/
│   ├── week1_problem_formulation.txt
│   ├── week2_data_acquisition.txt
│   ├── week3_data_acquisition_II.txt
│   ├── week4_data_exploration.txt
│   ├── week5_reflection.txt
│   ├── week6_data_wrangling.txt
│   ├── week7_feature_engineering.txt
│   ├── week8_modeling_classifiers.txt
│   ├── week9_clustering_finalization.txt
│   └── week10_dashboard_reflection.txt
│
├── data_dictionary.pdf
├── README.md
└── requirements.txt
```

The `data/` folder is in `.gitignore` because the raw AMI corpus is around 12 GB. Everything regenerates from scratch by running the notebooks in order — more on that below.

---

## Milestone 2 Breakdown

### data_wrangling.ipynb

This is where most of the actual thinking happened. The raw transcripts needed a lot of work before they were useful.

First we stripped AMI noise tokens — things like `<vocalsound>` and `<gap>` that the corpus annotators left in. Then we labeled every utterance as either a decision, an action item, or general discussion using regex patterns. We went with regex over a pretrained model because it's fully reproducible without any API dependencies, and it's easy to audit when something looks wrong.

The trickiest part was the redundancy score. We used bigram cosine similarity — for each utterance we check how similar it is to everything said earlier in the same meeting, and flag it if the match is above 0.30. Getting that threshold right took some manual checking.

All outputs are validated before saving — range checks, null checks, schema checks. If something breaks upstream, the notebook will tell you clearly instead of silently producing garbage.

### data_modeling.ipynb

We compared five classifiers on the utterance labeling task: Logistic Regression, Naive Bayes, Linear SVM, Random Forest, and XGBoost. They all get the same TF-IDF feature matrix so the comparison is fair — the only difference is the algorithm.

Because discussion utterances make up roughly 85% of the data, accuracy is a misleading metric. A model that predicts "discussion" for everything would score 85% and be completely useless. We use macro-F1 instead, and all models use class balancing so the minority classes actually get learned.

We also clustered the 171 meetings using K-Means on their efficiency metrics. The silhouette score pointed to 4 clusters, which ended up mapping nicely onto four tiers of meeting quality. The PCA scatter plot in the notebook shows the clusters are reasonably well-separated.

### data_visualization_static.ipynb

The dashboard has two parts. The first six panels are static and tell a complete story top to bottom without needing any interaction. The last four sections are widgets — dropdowns, sliders, toggles — that let you dig into specific meetings or experiment with the efficiency formula.

Static panels:
- KPI summary cards
- Efficiency score histogram and cluster pie
- Per-metric KDE curves by cluster
- Cluster explorer with top/bottom meeting tables
- Per-meeting deep dive (timeline, label counts, sample utterances)
- Model performance summary

Widget extensions:
- W1: Compare any two meetings side by side
- W2: Browse utterances from a meeting filtered by label
- W3: Change the efficiency formula weights and re-rank meetings live
- W4: See how each speaker contributed to decisions, actions, and discussion

---

## Running It

```bash
git clone https://github.com/asamr-ufl/CAP5771_Meeting-Transcript-Analyzer.git
cd CAP5771_Meeting-Transcript-Analyzer
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run the notebooks in this order — each one depends on the previous:

```
1. code/milestone1_analysis.ipynb
2. code/data_wrangling.ipynb
3. code/data_modeling.ipynb
4. code/data_visualization_static.ipynb
```

Step 1 is the slow one — it downloads the AMI corpus from Hugging Face (~12 GB of audio, stripped immediately after transcript extraction). If `data/raw/` already has transcript files it skips the download. Steps 2 through 4 are fast by comparison.

To open the dashboard:

```bash
cd code
jupyter notebook data_visualization_static.ipynb
```

Run all cells top to bottom. The interactive widgets need a live Jupyter kernel — they won't show up in a static HTML export. If you're on JupyterLab and the widgets aren't rendering:

```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

---

## Getting the Data

We can't include the raw corpus in the repo (12 GB). Here's how to get it back:

Run `milestone1_analysis.ipynb` — it pulls the text-only version of AMI from Hugging Face using `load_dataset("edinburghcstr/ami", "ihm")` and writes the transcript files to `data/raw/`. Audio is stripped immediately so you're not storing 12 GB long-term.

If you're on HiPerGator (UF research computing), the dataset is likely already cached and loads without a fresh download.

---

## What We Found

A few things stood out once the analysis was running.

Redundancy was by far the strongest signal separating good meetings from bad ones. High-efficiency meetings were not just ones with more decisions — they were ones where the same ground wasn't covered over and over. That was a bit surprising going in.

Action items were also consistently fuzzy. Even in moderately efficient meetings, the action clarity score was low. People would agree something needed to happen without naming who was doing it or when. That's a pattern worth flagging in a future version of the tool.

On the modeling side, XGBoost came out on top by macro-F1, but the gap between models wasn't huge. The harder problem isn't which model you pick — it's the boundary between action items and discussion, where the language is informal enough that even humans might not agree on the label.

---

## Team

- Rahul Reddy Asam — [@asamr-ufl](https://github.com/asamr-ufl)
- Ashruth Reddy Gangula — [@AshruthReddy1](https://github.com/AshruthReddy1)
- Rakesh Kumar Reddy Dodda — [@RAKESH-D0DDA](https://github.com/RAKESH-D0DDA)

