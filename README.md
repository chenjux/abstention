<!-- # The art of refusal: A survey of abstention in large language models -->
# Know Your Limits: A Survey of Abstention in Large Language Models
<!-- [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/chenjux/abstention)
[![Stars](https://img.shields.io/github/stars/yuzhimanhua/Awesome-Scientific-Language-Models?style=social)](https://img.shields.io/github/stars/yuzhimanhua/Awesome-Scientific-Language-Models?style=social)

[![Papers](https://img.shields.io/badge/PaperNumber-266-blue)](https://img.shields.io/badge/PaperNumber-266-blue)-->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRWelcome](https://img.shields.io/badge/PRs-Welcome-red)](https://img.shields.io/badge/PRs-Welcome-red)

# Abstention Methods in LLMs

## 1. Pretraining
- **Instruction Tuning**
- **Learning from Preferences**

## 2. Alignment
- **Instruction Tuning**
- **Learning from Preferences**

## 3. Inference

### Input-Processing
- **Query Processing**: Adapting queries to optimize abstention outcomes.
- **Probing LLM's Inner State**
- **Uncertainty Estimation**

### In-Processing
- **Calibration-Based**
- **Consistency-Based**
- **Prompting-Based**

### Output-Processing
- **Self-Evaluation**
- **LLM Collaboration**

The repository is part of our survey paper [**A Comprehensive Survey of Scientific Large Language Models and Their Applications in Scientific Discovery**] 
<!-- (https://arxiv.org/abs/2406.10833) and will be continuously updated. -->

**NOTE 1**: 

**NOTE 2**: 

**NOTE 3**: We appreciate contributions. If you have any suggested papers, feel free to reach out to chenjux@uw.edu. 





<p align="center">
    <img src="intro.svg" width="90%" style="align:center;"/>
</p>





## Contents
- [Pretraining](#pretraining)
- [Alignment](#alignment)
  - [Instruction Tuning](#alignment-instruction-tuning)
  - [Learning from Preferences](#alignment-learning-from-preferences)
- [Inference](#inference)
  - [Input-Processing](#inference-input-processing)
    - [Query Processing](#inference-query-processing)
    - [Probing LLM's Inner State](#inference-probing-llms-inner-state)
    - [Uncertainty Estimation](#inference-uncertainty-estimation)
  - [In-Processing](#inference-in-processing)
    - [Calibration-Based](#inference-calibration-based)
    - [Consistency-Based](#inference-consistency-based)
    - [Prompting-Based](#inference-prompting-based)
  - [Output-Processing](#inference-output-processing)
    - [Self-Evaluation](#inference-self-evaluation)
    - [LLM Collaboration](#inference-llm-collaboration)

---

## Pretraining
<h3 id="pretraining">Pretraining Methods</h3>

- **SciBERT**  
  _SciBERT: A Pretrained Model for Scientific Text_  
  **Conference**: EMNLP 2019  
  [[Paper](https://arxiv.org/abs/1903.10676)] [[GitHub](https://github.com/allenai/scibert)] [[Model (Base)](https://huggingface.co/allenai/scibert_scivocab_uncased)]

---

## Alignment
<h3 id="alignment-instruction-tuning">Instruction Tuning</h3>

- **Alignment for Honesty**  
  _Ensuring LLMs provide truthful responses by focusing on alignment._  
  **Conference**: NeurIPS 2024  
  [Paper](https://arxiv.org/abs/2312.07000) | [GitHub](https://github.com/GAIR-NLP/alignment-for-honesty)

- **R-tuning: Instructing Large Language Models to Say ‘I Don’t Know’**  
  _Introducing abstention strategies to handle knowledge gaps._  
  **Conference**: NAACL 2024  
  [Paper](https://aclanthology.org/2024.naacl-long.394/) | [GitHub](https://github.com/shizhediao/R-Tuning)

- **Keeping LLMs Aligned After Fine-Tuning: The Crucial Role of Prompt Templates**  
  _Exploring prompt design to maintain alignment post-fine-tuning._  
  **Conference**: ICLR 2024 R2-FM Workshop Poster  
  [Paper](https://arxiv.org/abs/2402.18540)

- **The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions**  
  _Building hierarchies to manage instruction priorities._  
  **Conference**: --  
  [Paper](https://arxiv.org/abs/2404.13208)

- **Don’t Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration**  
  _Leveraging multiple LLMs to recognize and manage knowledge gaps._  
  **Conference**: ACL 2024  
  [Paper](https://aclanthology.org/2024.acl-long.786.pdf) | [GitHub](https://github.com/BunsenFeng/AbstainQA)

- **Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization**  
  _Enhancing LLM defenses with prioritized goals._  
  **Conference**: ACL 2024  
  [Paper](https://aclanthology.org/2024.acl-long.481/) | [GitHub](https://github.com/thu-coai/JailbreakDefense_GoalPriority)

- **The Art of Saying No: Contextual Noncompliance in Language Models**  
  _Developing contextual strategies for noncompliance._  
  **Conference**: NeurIPS 2024 (Track: Datasets and Benchmarks Poster)  
  [Paper](https://www.arxiv.org/abs/2407.12043) | [GitHub](https://github.com/allenai/noncompliance)

- **Safety-Tuned LLaMAs: Lessons from Improving the Safety of Large Language Models That Follow Instructions**  
  _Lessons learned from safety tuning large-scale models._  
  **Conference**: ICLR 2024  
  [Paper](https://arxiv.org/abs/2309.07875) | [GitHub](https://github.com/vinid/safety-tuned-llamas)

- **The Art of Defending: A Systematic Evaluation and Analysis of LLM Defense Strategies on Safety and Over-Defensiveness**  
  _Systematic analysis of LLM safety and defensive strategies._  
  **Conference**: ACL 2024  
  [Paper](https://aclanthology.org/2024.findings-acl.776/)

<h3 id="alignment-learning-from-preferences">Learning from Preferences</h3>

- **RLHF (Reinforcement Learning with Human Feedback)**  
  _Fine-Tuning Language Models from Human Feedback_  
  **Conference**: NeurIPS 2020  
  [[Paper](https://arxiv.org/abs/2009.01325)] [[GitHub](https://github.com/openai/lm-human-preferences)] [[Model](https://huggingface.co/openai/gpt-3)]

---

## Inference

### Input-Processing
<h3 id="inference-query-processing">Query Processing</h3>

- **Example Method**: Query Optimization for Better Abstention  
  _Paper TBD_  
  [[Paper](https://linktothepaper.com)] [[GitHub](https://github.com/example/repo)] [[Model](https://huggingface.co/example)]

<h3 id="inference-probing-llms-inner-state">Probing LLM's Inner State</h3>

- **Example Method**: Probing for Confidence Levels  
  _Paper TBD_  
  [[Paper](https://linktothepaper.com)] [[GitHub](https://github.com/example/repo)] [[Model](https://huggingface.co/example)]

<h3 id="inference-uncertainty-estimation">Uncertainty Estimation</h3>

- **Method**: Bayesian Confidence Estimation  
  _Paper TBD_  
  [[Paper](https://linktothepaper.com)] [[GitHub](https://github.com/example/repo)] [[Model](https://huggingface.co/example)]

---

### In-Processing
<h3 id="inference-calibration-based">Calibration-Based</h3>

- **Method**: Confidence Calibration with Temperature Scaling  
  _ICML 2021_  
  [[Paper](https://arxiv.org/abs/1706.04599)] [[GitHub](https://github.com/example/repo)] [[Model](https://huggingface.co/example)]

<h3 id="inference-consistency-based">Consistency-Based</h3>

- **Method**: Consistency Across Prompts  
  _Paper TBD_  
  [[Paper](https://linktothepaper.com)] [[GitHub](https://github.com/example/repo)] [[Model](https://huggingface.co/example)]

<h3 id="inference-prompting-based">Prompting-Based</h3>

- **Method**: Strategic Prompt Engineering  
  _Paper TBD_  
  [[Paper](https://linktothepaper.com)] [[GitHub](https://github.com/example/repo)] [[Model](https://huggingface.co/example)]

---

### Output-Processing
<h3 id="inference-self-evaluation">Self-Evaluation</h3>

- **Method**: LLM Self-Critique for Output Validation  
  _Paper TBD_  
  [[Paper](https://linktothepaper.com)] [[GitHub](https://github.com/example/repo)] [[Model](https://huggingface.co/example)]

<h3 id="inference-llm-collaboration">LLM Collaboration</h3>

- **Method**: Multi-Model Collaboration for Abstention  
  _Paper TBD_  
  [[Paper](https://linktothepaper.com)] [[GitHub](https://github.com/example/repo)] [[Model](https://huggingface.co/example)]



## Citation
If you find this repository useful, please cite the following paper:
```
@article{zhang2024comprehensive,
  title={A Comprehensive Survey of Scientific Large Language Models and Their Applications in Scientific Discovery},
  author={Zhang, Yu and Chen, Xiusi and Jin, Bowen and Wang, Sheng and Ji, Shuiwang and Wang, Wei and Han, Jiawei},
  booktitle={EMNLP'24},
  pages={8783--8817},
  year={2024}
}
```
