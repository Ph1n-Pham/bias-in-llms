# bias-in-llms
Occupational Bias in Open-Source Pretrained Large Language Models: Analyzing Polarity towards Creative and Technical Professions

<!--- generate tag here: https://shields.io/badges --->
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/pragyy/datascience-readme-template?include_prereleases)
![GitHub last commit](https://img.shields.io/badge/last_commit-april_2024-blue)
![GitHub pull requests](https://img.shields.io/github/issues-pr/pragyy/datascience-readme-template)
![GitHub](https://img.shields.io/github/license/pragyy/datascience-readme-template)
![contributors](https://img.shields.io/github/contributors/pragyy/datascience-readme-template) 
![codesize](https://img.shields.io/github/languages/code-size/pragyy/datascience-readme-template) 

# Table of contents

- [Author](#author)
- [Table of contents](#table-of-contents)
- [Project Overview](#project-overview)
- [Installation and Setup](#installation-and-setup)
- [Data](#data)
- [Code structure](#code-structure)
- [Results and evaluation](#results-and-evaluation)
- [Future work](#future-work)
- [Acknowledgments and references](#acknowledgments-and-references)
- [License](#license)

# Author
Phineas Pham <br />
pham_p1@denison.edu <br />
Senior at Denison University majoring in Computer Science and Data Analytics.



# Project Overview

## Background:
As Large Language Models (LLMs) transform the tech industry, their integration into numerous applications raises concerns about potential biases. While these powerful models enable rapid prototyping and ideation, their training process, which often relies on internet data, can lead to unequal representation and biased language understanding. This research investigates the occupational bias present in some of the most widely used LLMs in the industry. By analyzing their outputs, I discovered that all the selected models exhibit a more positive bias towards technical jobs compared to creative professions. Notably, larger models tend to display greater occupational bias. Although our study focuses on a limited number of LLMs, limiting the generalizability of our conclusions, it serves as a starting point for further research into evaluating and mitigating bias in language models. Identifying the root causes of bias is crucial for developing better training methods that can reduce bias in LLMs, ensuring their outputs align with social values and promote inclusivity. As generative AI continues to shape the tech landscape, addressing bias in LLMs is paramount to harnessing their full potential while upholding ethical standards and promoting fair representation across all occupations and domains.

# Installation and Setup

## Codes and Resources Used
In this section I give user the necessary information about the software requirements.
- **Editor Used:**  Informing the user of the editor used to produce the project.
- **Python Version:** Informing the user of the version of python used for this project. If you are using some other language such as R, you can mention that as well.

## Python Packages Used
In this section, I include all the necessary dependencies needed to reproduce the project, so that the reader can install them before replicating the project. I categorize the long list of packages used as - 
- **General Purpose:** General purpose packages like `urllib, os, request`, and many more.
- **Data Manipulation:** Packages used for handling and importing dataset such as `pandas, numpy` and others.
- **Data Visualization:** Include packages which were used to plot graphs in the analysis or for understanding the ML modelling such as `seaborn, matplotlib` and others.
- **Machine Learning:** This includes packages that were used to generate the ML model such as `scikit, tensorflow`, etc.

The level of granularity you want to provide for the above list is entirely up to you. You can also add a few more levels, such as those for statistical analysis or data preparation, or you can simply incorporate them into the above list as is.

# Data

## Source Data and Acquisition
Our data source is from paper "BOLD: Dataset and metrics for measuring biases in open-ended language generation" (Dhamala et al., 2021). We acquire this data from [HuggingFace API](https://huggingface.co/datasets/AlexaAI/bold). 

## Data Preprocessing
To reproduce the data from this source to measure occupation language polarity, I split the profession prompts from the source into two groups: creative and technical occupations. More information on how I group these prompts can be viewed from the paper or in ```BOLD-dataset/profession_prompts```

# Code structure

The codebase of this project is structured as below:

```bash
├── BOLD-dataset/
│   ├── profession_prompts
│   │   ├── creative_occ_prompts.txt
│   │   ├── technical_occ_prompts.txt
│   ├── prompts
│   │   ├── gender_prompt.json
│   │   ├── political_ideology_prompt.json
│   │   ├── profession_prompt.json
│   │   ├── race_prompt.json
│   │   ├── religious_ideology_prompt.json
│   ├── wikipedia
│   │   ├── gender_wiki.json
│   │   ├── political_ideology_wiki.json
│   │   ├── profession_wiki.json
│   │   ├── race_wiki.json
│   │   ├── religious_ideology_wiki.json
│   ├── CODE_OF_CONDUCT.md
│   ├── CONTRIBUTING.md
│   ├── LICENSE.md
│   ├── README.md
├── regard_result/
│   ├── allenai_OLMo-1B_bias.txt
│   ├── allenai_OLMo-7B-Twin-2T_bias.txt
│   ├── allenai_OLMo-7B_bias.txt
│   ├── lmsys_vicuna-13b-v1.5_bias.txt
│   ├── lmsys_vicuna-7b-v1.5_bias.txt
│   ├── openlm-research_open_llama_13b_bias.txt
│   ├── openlm-research_open_llama_3b_v2_bias.txt
│   ├── openlm-research_open_llama_7b_v2_bias.txt
│   ├── tiiuae_falcon-7b_bias.txt
├── prompt.py
├── LICENSE
├── README.md
└── .gitignore
```

# Results and evaluation
Provide an overview of the results of your project, including any relevant metrics and graphs. Include explanations of any evaluation methodologies and how they were used to assess the quality of the model. You can also make it appealing by including any pictures of your analysis or visualizations.

# Future work
Outline potential future work that can be done to extend the project or improve its functionality. This will help others understand the scope of your project and identify areas where they can contribute.

# Acknowledgments and references
Acknowledge any contributors, data sources, or other relevant parties who have contributed to the project. This is an excellent way to show your appreciation for those who have helped you along the way.

I want to thank Dr. Supp and Denison Data Analytics Department for guidance and mentorship throughtout the project.

Ohio Supercomputer Center for providing computing services.

## References:

A Brief History of Large Language Models (LLM) | LinkedIn. (n.d.). Retrieved February 8, 2024, from https://www.linkedin.com/pulse/brief-history-large-language-models-llm-feiyu-chen/ <br />

Dong, X., Wang, Y., Yu, P. S., & Caverlee, J. (2023). Probing Explicit and Implicit Gender Bias through LLM Conditional Text Generation (arXiv:2311.00306). arXiv. https://doi.org/10.48550/arXiv.2311.00306 <br />

Explained: Neural networks. (2017, April 14). MIT News | Massachusetts Institute of Technology. https://news.mit.edu/2017/explained-neural-networks-deep-learning-0414 <br />

Gallegos, I. O., Rossi, R. A., Barrow, J., Tanjim, M. M., Kim, S., Dernoncourt, F., Yu, T., Zhang, R., & Ahmed, N. K. (2023). Bias and Fairness in Large Language Models: A Survey (arXiv:2309.00770). arXiv. https://doi.org/10.48550/arXiv.2309.00770<br />


Gupta, M. (2024, February 1). What are LLMs? Understanding different LLM families. Data Science in Your Pocket. https://medium.com/data-science-in-your-pocket/what-are-llms-understanding-different-llm-families-48b030c2e4fb<br />


Large language model. (2024). In Wikipedia. https://en.wikipedia.org/w/index.php?title=Large_language_model&oldid=1204963851<br />


Liang, P. P., Wu, C., Morency, L.-P., & Salakhutdinov, R. (2021). Towards Understanding and Mitigating Social Biases in Language Models. Proceedings of the 38th International Conference on Machine Learning, 6565–6576. https://proceedings.mlr.press/v139/liang21a.html<br />


Marr, B. (n.d.). A Short History Of ChatGPT: How We Got To Where We Are Today. Forbes. Retrieved February 9, 2024, from https://www.forbes.com/sites/bernardmarr/2023/05/19/a-short-history-of-chatgpt-how-we-got-to-where-we-are-today/<br />


Nadeem, M., Bethke, A., & Reddy, S. (2020). StereoSet: Measuring stereotypical bias in pretrained language models (arXiv:2004.09456). arXiv. https://doi.org/10.48550/arXiv.2004.09456<br />


Nangia, N., Vania, C., Bhalerao, R., & Bowman, S. R. (2020). CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models (arXiv:2010.00133). arXiv. https://doi.org/10.48550/arXiv.2010.00133<br />


Open Source Licenses: Types and Comparison. (n.d.). Snyk. Retrieved February 9, 2024, from https://snyk.io/learn/open-source-licenses/


# License
The License used is [MIT License](https://opensource.org/license/mit/).
