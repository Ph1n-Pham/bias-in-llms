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

# Quickstart

This repository contains the codebase to reproduce the results in the paper.

## Prerequisites

Before you begin, make sure you have the following installed:
  - Git
  - Conda (or any other Python environment manager)


#### Step 1: Clone the Repository

Open your terminal or command prompt and navigate to the directory where you want to clone the repository. Then, run the following command:
```
git clone https://github.com/Ph1n-Pham/bias-in-llms.git
```

#### Step 2: Create a Conda Environment

Next, create a new Conda environment and install the required dependencies. Navigate to the cloned repository and run the following commands:
```
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```
This will create a new Conda environment named myenv with Python 3.10 and install the required packages listed in the requirements.txt file.

#### Step 3: Run the Sample Script
Once the dependencies are installed, you can run the sample script prompt.py to generate text based on a given prompt and reproduce regard results for the predefined models. Navigate to the repository's root directory and run the following command:
```
python prompt.py
```

#### Contributing:
If you'd like to contribute to this project, please follow the standard GitHub workflow:

- Fork the repository
- Create a new branch (git checkout -b feature/your-feature)
- Commit your changes (git commit -am 'Add some feature')
- Push to the branch (git push origin feature/your-feature)
- Create a new Pull Request

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
This research serves as a starting point for further investigations into the root causes of bias in language models and the development of strategies to build more equitable and socially responsible AI systems. By fostering interdisciplinary collaborations between computer scientists, social scientists, and domain experts, we can work towards creating language models that truly reflect the diversity and richness of human experiences, free from the constraints of historical biases and prejudices. Ultimately, the goal should be to harness the immense potential of LLMs while ensuring that their outputs align with societal values of fairness, inclusivity, and respect for all individuals and communities, regardless of their chosen profession or creative pursuits.


# Acknowledgments and references: 
I would like to express my deepest appreciation to Dr. Sarah Supp and Dr. Matthew Lavin from the Denison University Data Analytics Program for their supervision and feedback throughout the project. Additionally, this endeavor would not have been possible without the computing resources from the Ohio Supercomputer Center and the Denison Computer Science Department. 

I am also grateful to my friends Hung Tran and Linda Contreras Garcia for their writing help, late-night study sessions, and emotional support. Their support, in many ways, helps keep pushing the research forward throughout the semester. 

Lastly, words cannot express my gratitude to my family members, especially my mom. Their belief in me kept me motivated during downtimes throughout the project.


## References:

A Brief History of Large Language Models (LLM) | LinkedIn. (n.d.). Retrieved February 8, 2024, from https://www.linkedin.com/pulse/brief-history-large-language-models-llm-feiyu-chen/ <br />

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.<br />

OpenAI. (2022, November 30). ChatGPT: Optimizing Language Models for Dialogue. https://openai.com/blog/chatgpt/<br />

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.<br />

Khashabi, D., Khot, T., Sabharwal, A., Clark, P., Etzioni, O., & Roth, D. (2020). Unifiedqa: Crossing format boundaries with a single qa system. arXiv preprint arXiv:2005.00700.<br />

Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the dangers of stochastic parrots: Can language models be too big?. Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency.<br />

Sheng, E., Chang, K. W., Natarajan, P., & Peng, N. (2019). The woman worked as a babysitter: On biased word embeddings. arXiv preprint arXiv:1905.09866.<br />

Liang, P. P., Wu, C., Baral, C., & Tian, Y. (2022). Towards understanding and mitigating social biases in language models. arXiv preprint arXiv:2202.08918.<br />

Dinan, E., Roller, S., Shuster, K., Fan, A., Boureau, Y. L., & Weston, J. (2019). Wizard of wikipedia: Knowledge-powered conversational agents. arXiv preprint arXiv:1811.01241.<br />

Rosenblatt, F. (1958). The perceptron: a probabilistic model for information storage and organization in the brain. Psychological review, 65(6), 386.<br />

McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. The bulletin of mathematical biophysics, 5(4), 115-133.<br />

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. arXiv preprint arXiv:2005.14165.<br />

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683.<br />

Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150.<br />

Free Software Foundation. (2007). GNU General Public License. https://www.gnu.org/licenses/gpl-3.0.en.html<br />

Massachusetts Institute of Technology. (n.d.). The MIT License. https://opensource.org/licenses/MIT<br />

Blodgett, S. L., Barocas, S., Daumé III, H., & Wallach, H. (2020). Language (technology) is power: A critical survey of "bias" in NLP. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (pp. 5454-5476).<br />

Bolukbasi, T., Chang, K. W., Zou, J. Y., Saligrama, V., & Kalai, A. T. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. Advances in neural information processing systems, 29.<br />

OpenLM-Research. (2023). Open_llama [GitHub repository]. https://github.com/openlm-research/open_llama<br />

Groeneveld, D., Kharitonov, E., Hanina, A., Sharir, O., Patu, K., Majumder, O., ... & Shoham, Y. (2024). OLMo: Open Language Models. arXiv preprint arXiv:2304.01256.<br />

Lowe, R., Ananyeva, M., Blackwood, R., Chmait, N., Foley, J., Hsu, M., ... & Zellers, R. (2023). Vicuna: An Open-Source Chatbot Impressing Humans in the Wild. arXiv preprint arXiv:2303.09592.<br />

Almazrouei, M., Elhajj, I. H., Alqudah, A., Alqudah, A., & Alsmadi, I. (2023). Falcon: A 180 Billion Parameter Open-Source Language Model. arXiv preprint arXiv:2304.07142.<br />

Tan, S., Tunuguntla, D., & van der Wal, O. (2022). You reap what you sow: On the Challenges of Bias Evaluation Under Multilingual Settings. https://openreview.net/forum?id=rK-7NhfSIW5<br />

Tatman, R. (2017). Gender and Dialect Bias in YouTube's Automatic Captions. Proceedings of the First ACL Workshop on Ethics in Natural Language Processing, 53-59. https://doi.org/10.18653/v1/W17-1606<br />

Tripodi, F. (2023). Ms. Categorized: Gender, notability, and inequality on Wikipedia. New Media & Society, 25(7), 1687-1707.<br />

Turpin, M., Michael, J., Perez, E., & Bowman, S.R. (2023). Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. arXiv:2305.04388<br />

U.S. Bureau of Labor Statistics. (2022). Employed persons by detailed occupation, sex, race, and Hispanic or Latino ethnicity. 
https://www.bls.gov/cps/cpsaat11.htm<br />

Vanmassenhove, E., Hardmeier, C., & Way, A. (2018). Getting Gender Right in Neural Machine Translation. Proceedings of EMNLP 2018, 3003-3008. 
https://doi.org/10.18653/v1/D18-1334<br />

Venkit, P.N., Gautam, S., Panchanadikar, R., Huang, T.H., & Wilson, S. (2023). Nationality Bias in Text Generation. arXiv:2302.02463<br />

Venkit, P.N., Srinath, M., & Wilson, S. (2022). A Study of Implicit Bias in Pretrained Language Models against People with Disabilities. Proceedings of COLING 2022, 1324-1332.<br />

Dhamala, J., Sun, T., Kumar, V., Krishna, S., Pruksachatkun, Y., Chang, K.-W., & Gupta, R. (2021). BOLD: Dataset and metrics for measuring biases in open-ended language generation. Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, 862-872. https://doi.org/10.1145/3442188.3445924<br />

Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334), 183-186.<br />

Sap, M., Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). The risk of racial bias in hate speech detection. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 1668-1678).<br />

Sheng, E., Chang, K. W., Natarajan, P., & Peng, N. (2021). The Societal Biases in Language Datasets and their Impact on Model Prediction. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 1711-1724).<br />

Mukaka, M. M. (2012). A guide to appropriate use of correlation coefficient in medical research. Malawi Medical Journal, 24(3), 69-71.<br />

Sedgwick, P. (2012). Pearson's correlation coefficient. BMJ, 345, e4483. https://doi.org/10.1136/bmj.e4483<br />

Taylor, R. (1990). Interpretation of the correlation coefficient: A basic review. Journal of Diagnostic<br />

Medical Sonography, 6(1), 35-39. 
https://doi.org/10.1177/875647939000600106<br />

Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values: Context, process, and purpose. The American Statistician, 70(2), 129-133. 
https://doi.org/10.1080/00031305.2016.1154108<br />




# License
This project is licensed under the [MIT License](https://opensource.org/license/mit/).
