<!-- # The art of refusal: A survey of abstention in large language models -->
# Know Your Limits: A Survey of Abstention in Large Language Models
<!-- [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/chenjux/abstention)
[![Stars](https://img.shields.io/github/stars/yuzhimanhua/Awesome-Scientific-Language-Models?style=social)](https://img.shields.io/github/stars/yuzhimanhua/Awesome-Scientific-Language-Models?style=social)

[![Papers](https://img.shields.io/badge/PaperNumber-266-blue)](https://img.shields.io/badge/PaperNumber-266-blue)-->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRWelcome](https://img.shields.io/badge/PRs-Welcome-red)](https://img.shields.io/badge/PRs-Welcome-red)

# Abstention Methods in LLMs

## 1. Pretraining
- **Data augmentation**

## 2. Alignment
- **Instruction Tuning**
- **Learning from Preferences**

## 3. Inference

### Input-Processing
- **Query Processing**: Adapting queries to optimize abstention outcomes.

### In-Processing
- **Probing LLM's Inner State**
- **Uncertainty Estimation**
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
  - [Data augmentation](#data-augmentation)
- [Alignment](#alignment)
  - [Instruction Tuning](#alignment-instruction-tuning)
  - [Learning from Preferences](#alignment-learning-from-preferences)
- [Inference](#inference)
  - [Input-Processing](#inference-input-processing)
    - [Query Processing](#inference-query-processing)
  - [In-Processing](#inference-in-processing)
    - [Probing LLM’s inner state](#inference-probing-llm’s-inner-state)
    - [Uncertainty estimation](#inference-uncertainty-estimation)
    - [Calibration-Based](#inference-calibration-based)
    - [Consistency-Based](#inference-consistency-based)
    - [Prompting-Based](#inference-prompting-based)
  - [Output-Processing](#inference-output-processing)
    - [Self-Evaluation](#inference-self-evaluation)
    - [LLM Collaboration](#inference-llm-collaboration)

---

## Pretraining
<h3 id="pretraining">Data augmentation</h3>

- **SciBERT**  
  _SciBERT: A Pretrained Model for Scientific Text_  
  **Conference**: EMNLP 2019  
  [[Paper](https://arxiv.org/abs/1903.10676)] [[GitHub](https://github.com/allenai/scibert)] [[Model (Base)](https://huggingface.co/allenai/scibert_scivocab_uncased)]

---

## Alignment
<h3 id="alignment">Instruction Tuning</h3>

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

- **Self-alignment for factuality**  
  _Mitigating hallucinations in LLMs via self-evaluation._  
  **Conference**: ACL 2024  
  [Paper](https://arxiv.org/abs/2402.09267) | [GitHub](https://github.com/zhangxy-2019/Self-Alignment-for-Factuality)

- **Can AI assistants know what they don’t know?**  
  _Exploring AI's understanding of its limitations._  
  **Conference**: ICML 2024 (Poster)  
  [Paper](https://arxiv.org/pdf/2401.13275) | [GitHub](https://github.com/OpenMOSS/Say-I-Dont-Know)

- **Learning to trust your feelings**  
  _Leveraging self-awareness in LLMs for hallucination mitigation._  
  **Conference**: ACL 2024  
  [Paper](https://arxiv.org/abs/2401.15449) | [GitHub](https://github.com/liangyuxin42/dreamcatcher)

- **Controllable preference optimization**  
  _Toward controllable multi-objective alignment._  
  **Conference**: EMNLP 2024  
  [Paper](https://aclanthology.org/2024.emnlp-main.85.pdf) | [GitHub](https://github.com/OpenBMB/CPO)

- **SafeRLHF**  
  _Safe reinforcement learning from human feedback._  
  **Conference**: ICLR 2024  
  [Paper](https://arxiv.org/pdf/2310.12773) | [GitHub](https://github.com/PKU-Alignment/safe-rlhf)

- **Training a helpful and harmless assistant with reinforcement learning from human feedback**  
  _Exploring the safe design of AI assistants._  
  **Conference**: ICLR 2024  
  [Paper](https://arxiv.org/abs/2310.12773) | [GitHub](https://github.com/PKU-Alignment/safe-rlhf)

- **Flame: Factuality-aware alignment for large language models**  
  _Addressing factuality in large language models._  
  **Conference**: NeurIPS 2024  
  [Paper](https://arxiv.org/abs/2405.01525) | [GitHub](https://github.com/Flame/Alignment)

- **Safe RLHF**  
  _Safe reinforcement learning from human feedback._  
  **Conference**: ICLR 2024  
  [Paper](https://arxiv.org/abs/2310.12773) | [GitHub](https://github.com/PKU-Alignment/safe-rlhf)

- **LLaMA: Open and efficient foundation language models**  
  _Designing open and efficient foundation models._  
  **Conference**: (No specific conference listed)  
  [Paper](https://arxiv.org/abs/2302.13971) | [GitHub](https://github.com/meta-llama/llama)

- **The art of saying no**  
  _Contextual noncompliance in language models._  
  **Conference**: NeurIPS 2024  
  [Paper](https://www.arxiv.org/abs/2407.12043) | [GitHub](https://github.com/allenai/noncompliance)

- **Defending against backdoor attacks in natural language generation**  
  _Addressing backdoor attacks in natural language models._  
  **Conference**: (No specific conference listed)  
  [Paper](https://arxiv.org/abs/2106.01810) | [GitHub](https://github.com/defend-backdoor-attacks)

- **Break the breakout**  
  _Reinventing LM defense against jailbreak attacks with self-refinement._  
  **Conference**: AAAI-23  
  [Paper](https://tianweiz07.github.io/Papers/23-aaai.pdf) | [GitHub](https://github.com/self-refinement-defense)

---

## Inference

### Input-Processing
<h4 id="inference-query-processing">Query Processing</h4>

- **Selectively answering ambiguous questions**  
  _Designing models to handle ambiguity in natural language understanding._  
  **Conference**: EMNLP 2023  
  [Paper](https://arxiv.org/abs/2305.14613)  

- **ONION: A simple and effective defense against textual backdoor attacks**  
  _A method to defend against textual backdoor attacks in NLP models._  
  **Conference**: EMNLP 2021  
  [Paper](https://aclanthology.org/2021.emnlp-main.752.pdf) | [GitHub](https://github.com/thunlp/ONION)

- **Token-level adversarial prompt detection based on perplexity measures and contextual information**  
  _Detecting adversarial prompts by analyzing token-level perplexity and context._  
  **Conference**: (No specific conference listed)  
  [Paper](https://arxiv.org/abs/2311.11509)

- **Defending against backdoor attacks in natural language generation**  
  _Addressing security issues in natural language generation models._  
  **Conference**: AAAI 2023  
  [Paper](https://arxiv.org/abs/2106.01810)

- **Defending pre-trained language models as few-shot learners against backdoor attacks**  
  _Protection strategies for pre-trained models against backdoor threats in few-shot settings._  
  **Conference**: NeurIPS 2023  
  [Paper](https://arxiv.org/abs/2309.13256) | [GitHub](https://github.com/zhaohan-xi/PLM-prompt-defense)

- **Baseline defenses for adversarial attacks against aligned language models**  
  _Establishing baseline strategies for defending aligned language models from adversarial attacks._  
  **Conference**: (No specific conference listed)  
  [Paper](https://arxiv.org/abs/2309.00614)

- **Bddr: An effective defense against textual backdoor attacks**  
  _A defense mechanism for protecting NLP models from textual backdoor attacks._  
  **Conference**: (No specific conference listed)  
  [Paper](https://dl.acm.org/doi/abs/10.1016/j.cose.2021.102433)

- **Certifying LLM safety against adversarial prompting**  
  _Methods for certifying the safety of large language models against adversarial prompts._  
  **Conference**: (No specific conference listed)  
  [Paper](https://arxiv.org/abs/2309.02705)

- **Build it break it fix it for dialogue safety: Robustness from adversarial human attack**  
  _Improving dialogue model safety by simulating adversarial human attacks._  
  **Conference**: EMNLP 2019  
  [Paper](https://arxiv.org/abs/1908.06083)

---

### In-Processing

<h3 id="inference-probing-llm’s-inner-state">Probing LLM’s inner state</h3>

- **Method**: Confidence Calibration with Temperature Scaling  
  _ICML 2021_  
  [[Paper](https://arxiv.org/abs/1706.04599)] [[GitHub](https://github.com/example/repo)] [[Model](https://huggingface.co/example)]
  
<h3 id="inference-uncertainty-estimation">Uncertainty estimation</h3>

- **Method**: Confidence Calibration with Temperature Scaling  
  _ICML 2021_  
  [[Paper](https://arxiv.org/abs/1706.04599)] [[GitHub](https://github.com/example/repo)] [[Model](https://huggingface.co/example)]
  
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
