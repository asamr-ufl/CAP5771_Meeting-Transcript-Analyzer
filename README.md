Meeting Transcript Efficiency Analyzer
CAP5771 – Intro to Data Science
Milestone 1

**Problem Overview
**

This project aims to measure whether meetings are actually productive. Instead of relying on opinions, we analyze meeting transcripts to understand how often decisions are made, whether action items are clearly defined, and how much repetitive discussion occurs. The goal is to use measurable indicators to evaluate meeting effectiveness in a consistent and objective way.

**Current Practice and Limitations
**
Most meetings are assessed informally through personal impressions or short feedback surveys. There is no standardized method to analyze what was actually said during a meeting. As a result, productivity is often judged subjectively, and transcripts, when available, are rarely examined in a structured way. This project addresses that gap by turning conversation data into quantifiable metrics.

**Data Source
**
The dataset used is the AMI Meeting Corpus, accessed through the Hugging Face datasets library. It contains 171 real workplace meetings and over 134,000 utterances. Each record includes a meeting identifier, speaker identifier, timestamps, and transcribed text. Audio data is excluded to focus on text-based analysis.

**Data Reproducibility
**
All data is publicly available. The corpus is downloaded programmatically, and extracted transcript files are stored in the data/raw directory. Processed datasets created during analysis are saved in data/processed. No proprietary data is used.

**Derived Variables
**
Additional features are created during preprocessing, including word counts, meeting duration, number of speakers, total utterances, and speaking rate. These variables support exploratory analysis and form the foundation for future metrics such as decision density and redundancy scoring.

**Repository Structure
**
The repository includes the main Jupyter notebook with all analysis steps, a SQLite database containing structured transcript data, a schema image describing table relationships, a data dictionary, diary documentation for each milestone stage, and a requirements file for reproducibility.

**Reproduction Steps
**
To reproduce the project, clone the repository, create a virtual environment, install dependencies from requirements.txt, and run the notebook sequentially. The notebook loads the data, performs exploratory analysis, generates visualizations, and builds the database.

This milestone establishes the problem definition, data acquisition process, database structure, and exploratory insights needed for the next stages of the project.
