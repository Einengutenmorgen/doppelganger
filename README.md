# Digital Doppelgangers
### Synthetic Social Media Post Generation of real Users

![titleImage](https://github.com/Einengutenmorgen/doppelganger/blob/main/title_image.png)

## Overview

This project explores the generation of synthetic social media posts using language models, focusing on persona-based content creation, evaluation, and fine-tuning. The primary goal is to create realistic and representative social media content for research purposes. Experiments were conducted using both closed-source (GPT-4o) and open-source (Llama 3.1 70B) language models.

## Experimental Setup

The project follows an iterative pipeline:

1.  **Persona Creation:** Generate personas with specific categories.
2.  **Post Generation:** Create synthetic social media posts for each persona based on their characteristics.
3.  **Evaluation:** Evaluate the generated posts for quality, style, and relevance.
4.  **Fine-tuning:** Fine-tune the Llama 3.1 70B model to improve the realism and representativeness of the generated posts.

## Scripts

*   `bin/create_persona_batch`: Creates batches of personas with randomly selected categories.
*   `bin/create_posts`: Generates synthetic social media posts for each persona.
*   `bin/evaluate`: Evaluates the generated posts, potentially using an LLM-based judge.
*   `notebooks/second_finetuning.ipynb`: Fine-tunes the Llama 3.1 70B model using the generated personas and posts.
*   `bin/final_pipeline`: Orchestrates the entire pipeline, including persona creation, post generation, evaluation, and fine-tuning.
*   `bin/batch_eval_2`: Performs batch evaluation of the generated content, likely after fine-tuning.

## Data

The project utilizes data from multiple CSV files located in the `results/Final_finetune` directory. The `notebooks/second_finetuning.ipynb` notebook combines and preprocesses this data for fine-tuning.

## Usage

1.  **Persona Creation:**
    ```bash
    ./bin/create_persona_batch [arguments]
    ```

2.  **Post Generation:**
    ```bash
    ./bin/create_posts [arguments]
    ```

3.  **Evaluation:**
    ```bash
    ./bin/evaluate [arguments]
    ```

4.  **Fine-tuning:**
    Run the `notebooks/second_finetuning.ipynb` notebook.

5.  **Final Pipeline:**
    ```bash
    ./bin/final_pipeline [arguments]
    ```

6.  **Batch Evaluation:**
    ```bash
    ./bin/batch_eval_2 [arguments]
    ```

Refer to the individual script documentation or help messages for specific arguments and options.

## Cost Analysis

The initial iteration using GPT-4o cost EUR 66.96 for 28,000 API requests. This information can be used to estimate the cost of future runs and compare the cost-effectiveness of different language models.

## Notes

*   The project aims to generate synthetic social media posts that resemble real-world content.
*   Fine-tuning is performed to address divergences observed in the initial GPT-4o generated posts, such as orthography, punctuation, content offensiveness, hashtag usage and post length.
*   The dataset is partitioned into training and evaluation sets to mitigate overfitting during fine-tuning.

--- automaticaly generated read.me ---

