# Replication Package: Security Concerns in GitHub Copilot

This repository contains all files needed to replicate the key steps in the master's thesis:
**"Security Concerns in Generative AI Coding Assistants."**

## Structure

- `final_dataset.csv`: Filtered dataset of 400 security-relevant comments (Reddit, SO, HN) used for topic modeling.
- `codebook.docx`: Thematic codebook used in open coding and analysis.
- `bertopic_pipeline.py`: Python code used to perform BERTopic modeling (with tuned parameters).
- `manual_validation_prompt.docx`: ChatGPT prompt used during manual comment validation.
- `sentiment_analysis.py`: Script used to compute sentiment scores and create the final heatmap.
- `topic_assignment.csv`: Dataset showing comment-to-topic assignments after topic modeling.

## How to Use

1. Reuse `manual_validation_prompt.txt` to understand the validation procedure.
2. Run `bertopic_pipeline.py` to generate topic clusters.
3. Use `topic_assignment.csv` as the mapping for analysis.
4. Refer to `codebook.docx` for qualitative coding categories.
5. Execute `sentiment_analysis.py` to produce sentiment scores and heatmaps.
 

## Notes

- Raw scraping scripts are excluded to avoid GitHub policy violations.
- The dataset provided was filtered using curated security keywords.
- All personally identifiable or sensitive data has been removed.

## Citation

If you use this material, please cite the original thesis or contact the author.
