# bias-in-llms
Measuring bias in open-source pretrained large language model: A study on the relationship between trained data characteristics and model size on language models’ bias 

<!--- generate tag here: https://shields.io/badges --->
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/pragyy/datascience-readme-template?include_prereleases)
![GitHub last commit](https://img.shields.io/badge/last_commit-april_2024-blue)
![GitHub pull requests](https://img.shields.io/github/issues-pr/pragyy/datascience-readme-template)
![GitHub](https://img.shields.io/github/license/pragyy/datascience-readme-template)
![contributors](https://img.shields.io/github/contributors/pragyy/datascience-readme-template) 
![codesize](https://img.shields.io/github/languages/code-size/pragyy/datascience-readme-template) 

# Author
Phineas Pham <br />
pham_p1@denison.edu

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

# Project Overview

## Background:
Large language models (LLMs), products of the ongoing evolution in deep learning and neural networks, have emerged as prominent entities within the technological sphere. While language models have undergone years of development, a pivotal moment occurred in 2017 when Google introduced its groundbreaking research paper titled "Attention is All You Need," (A Brief History of Large Language Models (LLM) | LinkedIn, n.d.) alongside the revolutionary Transformer architecture (Gupta, 202). This innovative architecture laid the groundwork for the creation of one of the most renowned LLM applications, ChatGPT, which debuted in June 2020 (Marr, n.d.). Since then, the ever-expanding capabilities of LLMs have facilitated the development of a plethora of fascinating applications spanning diverse domains, including language generation, translation, question-answering, and summarization. However, with the widespread adoption of LLMs and their impressive performance, concerns have arisen regarding their potential to perpetuate and exacerbate social biases present within their output. Consequently, bias has emerged as one of the most significant social issues associated with LLMs. In response to this challenge, researchers have redirected their focus toward designing frameworks and solutions aimed at quantifying and mitigating social bias in LLMs. This research endeavors to address this critical issue, necessitating a comprehensive understanding of how both data characteristics and model size influence the bias present in LLMs.

# Installation and Setup

In this section, provide detailed instructions on how to set up the project on a local machine. This includes any necessary dependencies, software requirements, and installation steps. Make sure to include clear and concise instructions so that others can easily replicate your setup.

I like to structure it as below - 


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

## Source Data
StereoSet

## Data Acquisition

HuggingFace

## Data Preprocessing
Acquired data is not always squeaky clean, so preprocessing them are an integral part of any data analysis. In this section you can talk about the same.

# Code structure

Here is the basic suggested skeleton for your data science repo (you can structure your repository as needed ):

```bash
├── data
│   ├── data1.csv
│   ├── data2.csv
│   ├── cleanedData
│   │   ├── cleaneddata1.csv
|   |   └── cleaneddata2.csv
├── data_acquisition.py
├── data_preprocessing.ipynb
├── data_analysis.ipynb
├── data_modelling.ipynb
├── Img
│   ├── img1.png
│   ├── Headerheader.jpg
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
