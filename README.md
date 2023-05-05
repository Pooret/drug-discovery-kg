# Drug Discovery Knowledge Graph Construction and Information Extraction

This project aims to accelerate drug discovery and development by constructing a comprehensive knowledge graph and extracting relevant information from biomedical text data. Using state-of-the-art natural language processing (NLP) techniques, the project involves tasks such as named entity recognition, relation extraction, text classification, and multi-task learning. The ultimate goal is to facilitate a better understanding of drug candidates, targets, mechanisms of action, and clinical trial phases, thereby reducing costs and increasing efficiency in the pharmaceutical industry.

## Table of Contents
- [Overview](#overview)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Results and Evaluation](#results-and-evaluation)
- [License](#license)

## Overview

The project focuses on the following tasks and components:

1. Data collection and preprocessing: Gather data from sources like PubMed, ClinicalTrials.gov, patent databases, and scientific literature repositories. Clean and preprocess the data to ensure consistency and quality.
2. Information extraction: Implement NLP models for named entity recognition and relation extraction to identify relevant entities and relationships in the text data.
3. Text classification: Train and evaluate text classification models to categorize extracted information into appropriate classes.
4. Multi-task learning: Develop models for simultaneous information extraction and text classification tasks, optimizing overall performance.
5. Distant supervision: Use distant supervision techniques to create a large-scale annotated dataset for relation extraction and improve model performance.
6. Knowledge graph construction: Integrate extracted entities and relationships into a structured knowledge graph, enabling researchers to analyze and mine insights.

## Data Sources

This project utilizes data from various sources, including:

- PubMed: Biomedical literature database
- ClinicalTrials.gov: Clinical trial registry and results database
- Patent databases: Databases containing drug-related patents
- Drug databases: Databases containing information on drug candidates and their properties

## Project Structure

- `docs`: Project overview, flowchart, data and model descriptions, and presentation slides
- `models`: Trained models for named entity recognition, relation extraction, and text classification
- `notebooks`: Jupyter notebooks for data exploration, model training, and evaluation
- `src`: Source code for data preprocessing, feature extraction, model training, evaluation, and knowledge graph construction
- `results`: Model outputs, such as evaluation metrics, charts, and intermediate results

## Results and Evaluation

The project includes a detailed evaluation of the models and techniques used in the various tasks. The evaluation metrics, benchmarks, and visualizations.

