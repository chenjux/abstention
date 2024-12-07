<!-- # The art of refusal: A survey of abstention in large language models -->
# Know Your Limits: A Survey of Abstention in Large Language Models
<!-- [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/chenjux/abstention)
[![Stars](https://img.shields.io/github/stars/yuzhimanhua/Awesome-Scientific-Language-Models?style=social)](https://img.shields.io/github/stars/yuzhimanhua/Awesome-Scientific-Language-Models?style=social)

[![Papers](https://img.shields.io/badge/PaperNumber-266-blue)](https://img.shields.io/badge/PaperNumber-266-blue)-->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRWelcome](https://img.shields.io/badge/PRs-Welcome-red)](https://img.shields.io/badge/PRs-Welcome-red)

# Abstention Methods & Benchmarks in LLMs

<!--The repository is part of our survey paper [**Know Your Limits: A Survey of Abstention in Large Language Models**](https://arxiv.org/abs/2407.18418) and will be continuously updated.-->

**NOTE 1**: The current order of papers follows the sequence in which they appear in our survey paper. Future papers will be added in chronological order.

<!--**NOTE 2**: We appreciate contributions. If you have any suggested papers, feel free to reach out to [chenjux@uw.edu](mailto:chenjux@uw.edu) or submit a pull request.-->

<p align="center">
    <img src="abstention.png" width="90%" style="align:center;"/>
</p>





## Contents
- [Pretraining](#pretraining)

- [Alignment](#alignment)
  - [Supervised Finetuning](#alignment-instruction-tuning)
  - [Learning from Preferences](#alignment-learning-from-preferences)
- [Inference](#inference)
  - [Input-Processing](#inference-input-processing)
    - [Query Processing](#inference-query-processing)
  - [In-Processing](#inference-in-processing)
    - [Probing LLM’s inner state](#inference-probing-llms-inner-state)
    - [Uncertainty estimation](#inference-uncertainty-estimation)
    - [Calibration-Based](#inference-calibration-based)
    - [Consistency-Based](#inference-consistency-based)
    - [Prompting-Based](#inference-prompting-based)
  - [Output-Processing](#inference-output-processing)
    - [Self-Evaluation](#inference-self-evaluation)
    - [LLM Collaboration](#inference-llm-collaboration)

- [Abstention evaluation benchmarks](#abstention-evaluation-benchmarks)
  - [Query](#query)
  - [Model](#model)
  - [Human Values](#human-values)
  - [General](#general)
---

## Pretraining

## Alignment
<h3 id="alignment">Supervised Finetuning</h3>

- **Alignment for Honesty**  
  **Conference**: NeurIPS 2024  
  [Paper](https://arxiv.org/abs/2312.07000) | [GitHub](https://github.com/GAIR-NLP/alignment-for-honesty)

- **R-tuning: Instructing Large Language Models to Say ‘I Don’t Know’**  
  **Conference**: NAACL 2024  
  [Paper](https://aclanthology.org/2024.naacl-long.394/) | [GitHub](https://github.com/shizhediao/R-Tuning)

- **Keeping LLMs Aligned After Fine-Tuning: The Crucial Role of Prompt Templates**  
  **Conference**: ICLR 2024 R2-FM Workshop Poster  
  [Paper](https://arxiv.org/abs/2402.18540)

- **The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions**  
  **Conference**: --  
  [Paper](https://arxiv.org/abs/2404.13208)

- **Don’t Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration**  
  **Conference**: ACL 2024  
  [Paper](https://aclanthology.org/2024.acl-long.786.pdf) | [GitHub](https://github.com/BunsenFeng/AbstainQA)

- **Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization**  
  **Conference**: ACL 2024  
  [Paper](https://aclanthology.org/2024.acl-long.481/) | [GitHub](https://github.com/thu-coai/JailbreakDefense_GoalPriority)

- **The Art of Saying No: Contextual Noncompliance in Language Models**  
  **Conference**: NeurIPS 2024 (Track: Datasets and Benchmarks Poster)  
  [Paper](https://www.arxiv.org/abs/2407.12043) | [GitHub](https://github.com/allenai/noncompliance)

- **Safety-Tuned LLaMAs: Lessons from Improving the Safety of Large Language Models That Follow Instructions**  
  **Conference**: ICLR 2024  
  [Paper](https://arxiv.org/abs/2309.07875) | [GitHub](https://github.com/vinid/safety-tuned-llamas)

- **The Art of Defending: A Systematic Evaluation and Analysis of LLM Defense Strategies on Safety and Over-Defensiveness**  
  **Conference**: ACL 2024  
  [Paper](https://aclanthology.org/2024.findings-acl.776/)
  
---

<h3 id="alignment-learning-from-preferences">Learning from Preferences</h3>


- **Self-alignment for factuality**  
  **Conference**: ACL 2024  
  [Paper](https://arxiv.org/abs/2402.09267) | [GitHub](https://github.com/zhangxy-2019/Self-Alignment-for-Factuality)

- **Can AI assistants know what they don’t know?**  
  **Conference**: ICML 2024 (Poster)  
  [Paper](https://arxiv.org/pdf/2401.13275) | [GitHub](https://github.com/OpenMOSS/Say-I-Dont-Know)

- **Learning to trust your feelings**  
  **Conference**: ACL 2024  
  [Paper](https://arxiv.org/abs/2401.15449) | [GitHub](https://github.com/liangyuxin42/dreamcatcher)

- **Controllable preference optimization**  
  **Conference**: EMNLP 2024  
  [Paper](https://aclanthology.org/2024.emnlp-main.85.pdf) | [GitHub](https://github.com/OpenBMB/CPO)

- **SafeRLHF**  
  **Conference**: ICLR 2024  
  [Paper](https://arxiv.org/pdf/2310.12773) | [GitHub](https://github.com/PKU-Alignment/safe-rlhf)

- **Training a helpful and harmless assistant with reinforcement learning from human feedback**  
  **Conference**: ICLR 2024  
  [Paper](https://arxiv.org/abs/2310.12773) | [GitHub](https://github.com/PKU-Alignment/safe-rlhf)

- **Flame: Factuality-aware alignment for large language models**  
  **Conference**: NeurIPS 2024  
  [Paper](https://arxiv.org/abs/2405.01525) | [GitHub](https://github.com/Flame/Alignment)

- **Safe RLHF**  
  **Conference**: ICLR 2024  
  [Paper](https://arxiv.org/abs/2310.12773) | [GitHub](https://github.com/PKU-Alignment/safe-rlhf)

- **LLaMA: Open and efficient foundation language models**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2302.13971) | [GitHub](https://github.com/meta-llama/llama)

- **The art of saying no**  
  **Conference**: NeurIPS 2024  
  [Paper](https://www.arxiv.org/abs/2407.12043) | [GitHub](https://github.com/allenai/noncompliance)

- **Defending against backdoor attacks in natural language generation**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2106.01810) | [GitHub](https://github.com/defend-backdoor-attacks)

- **Break the breakout**   
  **Conference**: AAAI-23  
  [Paper](https://tianweiz07.github.io/Papers/23-aaai.pdf) | [GitHub](https://github.com/self-refinement-defense)

---

## Inference


<h3 id="inference-input-processing">Input-Processing</h3>
<h4 id="inference-query-processing">Query Processing</h4>

- **Selectively answering ambiguous questions**  
  **Conference**: EMNLP 2023  
  [Paper](https://arxiv.org/abs/2305.14613)  

- **ONION: A simple and effective defense against textual backdoor attacks**  
  **Conference**: EMNLP 2021  
  [Paper](https://aclanthology.org/2021.emnlp-main.752.pdf) | [GitHub](https://github.com/thunlp/ONION)

- **Token-level adversarial prompt detection based on perplexity measures and contextual information**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2311.11509)

- **Defending against backdoor attacks in natural language generation**  
  **Conference**: AAAI 2023  
  [Paper](https://arxiv.org/abs/2106.01810)

- **Defending pre-trained language models as few-shot learners against backdoor attacks**  
  **Conference**: NeurIPS 2023  
  [Paper](https://arxiv.org/abs/2309.13256) | [GitHub](https://github.com/zhaohan-xi/PLM-prompt-defense)

- **Baseline defenses for adversarial attacks against aligned language models**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2309.00614)

- **Bddr: An effective defense against textual backdoor attacks**  
  **Conference**: N/A  
  [Paper](https://dl.acm.org/doi/abs/10.1016/j.cose.2021.102433)

- **Certifying LLM safety against adversarial prompting**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2309.02705)

- **Build it break it fix it for dialogue safety: Robustness from adversarial human attack**  
  **Conference**: EMNLP 2019  
  [Paper](https://arxiv.org/abs/1908.06083)

---


<h3 id="inference-in-processing">In-Processing</h3>
<h4 id="inference-probing-llms-inner-state">Probing LLM’s inner state</h4>

- **Language models (mostly) know what they know**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/pdf/2207.05221)

- **The internal state of an LLM knows when it’s lying**  
  **Conference**: EMNLP 2023  
  [Paper](https://arxiv.org/abs/2304.13734)

- **Inferaligner: Inference-time alignment for harmlessness through cross-model guidance**  
  **Conference**: ACL 2024  
  [Paper](https://arxiv.org/abs/2401.11206) | [GitHub](https://github.com/Jihuai-wpy/InferAligner)

- **Simple and principled uncertainty estimation with deterministic deep learning via distance awareness**  
  **Conference**: NeurIPS 2020  
  [Paper](https://arxiv.org/abs/2006.10108) | [GitHub](https://github.com/google/uncertainty-baselines/tree/master/baselines)

- **INSIDE: LLMs’ internal states retain the power of hallucination detection**  
  **Conference**: ICLR 2024  
  [Paper](https://arxiv.org/abs/2402.03744) | [GitHub](https://github.com/alibaba/eigenscore)

- **Selective question answering under domain shift**  
  **Conference**: ACL 2020  
  [Paper](https://arxiv.org/abs/2006.09462)

- **The curious case of hallucinatory (un)answerability: Finding truths in the hidden states of overconfident large language models**  
  **Conference**: EMNLP 2023  
  [Paper](https://arxiv.org/abs/2310.11877) | [GitHub](https://github.com/lovodkin93/unanswerability)

- **Language models are Homer Simpson! Safety re-alignment of fine-tuned language models through task arithmetic**  
  **Conference**: ACL 2024  
  [Paper](https://arxiv.org/abs/2402.11746) | [GitHub](https://github.com/declare-lab/resta)

---

<h4 id="inference-uncertainty-estimation">Uncertainty estimation</h4>

- **Teaching models to express their uncertainty in words**  
  **Conference**: TMLR  
  [Paper](https://arxiv.org/abs/2205.14334)

- **Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models finetuned with human feedback**  
  **Conference**: EMNLP 2023  
  [Paper](https://arxiv.org/abs/2305.14975)

- **Uncertainty-based abstention in LLMs improves safety and reduces hallucinations**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2404.10960)

- **Language models (mostly) know what they know**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/pdf/2207.05221)

- **Shifting attention to relevance: Towards the uncertainty estimation of large language models**  
  **Conference**: ACL 2024  
  [Paper](https://arxiv.org/abs/2307.01379) | [GitHub](https://github.com/jinhaoduan/SAR)

- **Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs**  
  **Conference**: ICLR 2024  
  [Paper](https://arxiv.org/abs/2306.13063) | [GitHub](https://github.com/MiaoXiong2320/llm-uncertainty)

- **LLaMAs know what GPTs don’t show: Surrogate models for confidence estimation**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2311.08877)

- **GPT-4 technical report**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2303.08774)

- **Selectively answering ambiguous questions**  
  **Conference**: EMNLP 2023  
  [Paper](https://arxiv.org/abs/2305.14613)

- **Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation**  
  **Conference**: ICLR 2023  
  [Paper](https://arxiv.org/abs/2302.09664)

- **Self-evaluation improves selective generation in large language models**  
  **Conference**: NeurIPS 2023 Workshops  
  [Paper](https://arxiv.org/abs/2312.09300)

- **Relying on the unreliable: The impact of language models’ reluctance to express uncertainty**  
  **Conference**: ACL 2024  
  [Paper](https://arxiv.org/abs/2401.06730)

<h4 id="inference-calibration-based">Calibration-Based</h4>

- **Calibrating sequence likelihood improves conditional language generation**  
  **Conference**: ICLR 2023  
  [Paper](https://arxiv.org/abs/2210.00045)

- **Uncertainty quantification with pre-trained language models: A large-scale empirical analysis**  
  **Conference**: EMNLP 2022  
  [Paper](https://arxiv.org/abs/2210.04714)

- **How can we know when language models know? On the calibration of language models for question answering**  
  **Conference**: TACL 2021  
  [Paper](https://arxiv.org/abs/2012.00955) | [GitHub](https://github.com/jzbjyb/lm-calibration)

- **Decomposing uncertainty for large language models through input clarification ensembling**  
  **Conference**: ICML 2024  
  [Paper](https://arxiv.org/abs/2311.08718) | [GitHub](https://github.com/UCSB-NLP-Chang/llm_uncertainty)

- **Investigating selective prediction approaches across several tasks in IID, OOD, and adversarial settings**  
  **Conference**: ACL 2022  
  [Paper](https://arxiv.org/abs/2203.00211)

- **TyDi QA: A benchmark for information-seeking question answering in typologically diverse languages**  
  **Conference**: TACL 2020  
  [Paper](https://arxiv.org/abs/2003.05002)

- **Reducing conversational agents’ overconfidence through linguistic calibration**  
  **Conference**: TACL 2022  
  [Paper](https://arxiv.org/abs/2012.14983)

- **Learning confidence for transformer-based neural machine translation**  
  **Conference**: ACL 2022  
  [Paper](https://arxiv.org/abs/2203.11413) | [GitHub](https://github.com/yulu-dada/Learned-conf-NMT)

- **Batchensemble: An alternative approach to efficient ensemble and lifelong learning**  
  **Conference**: ICLR 2020  
  [Paper](https://arxiv.org/abs/2002.06715) | [GitHub](https://github.com/google/edward2)

- **On uncertainty calibration and selective generation in probabilistic neural summarization: A benchmark study**  
  **Conference**: EMNLP 2023  
  [Paper](https://arxiv.org/abs/2304.08653)

- **Calibration of pre-trained transformers**  
  **Conference**: EMNLP 2020  
  [Paper](https://arxiv.org/abs/2003.07892) | [GitHub](https://github.com/shreydesai/calibration)

- **LACIE: Listener-aware finetuning for confidence calibration in large language models**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2405.21028) | [GitHub](https://github.com/esteng/pragmatic_calibration)
  
<h4 id="inference-consistency-based">Consistency-Based</h4>

- **The curious case of hallucinatory (un)answerability: Finding truths in the hidden states of overconfident large language models**  
  **Conference**: EMNLP 2023  
  [Paper](https://arxiv.org/abs/2310.11877) | [GitHub](https://github.com/lovodkin93/unanswerability)

- **Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs**  
  **Conference**: ICLR 2024  
  [Paper](https://arxiv.org/abs/2306.13063) | [GitHub](https://github.com/MiaoXiong2320/llm-uncertainty)

- **Knowing what LLMs DO NOT know: A simple yet effective self-detection method**  
  **Conference**: ACL 2024  
  [Paper](https://arxiv.org/abs/2310.17918)

- **Generating with confidence: Uncertainty quantification for black-box large language models**  
  **Conference**: TMLR 2024  
  [Paper](https://arxiv.org/abs/2305.19187) | [GitHub](https://github.com/zlin7/UQ-NLG)

- **Defending large language models against jailbreak attacks via semantic smoothing**  
  **Conference**: ICLR 2024  
  [Paper](https://arxiv.org/abs/2402.16192) | [GitHub](https://github.com/UCSB-NLP-Chang/SemanticSmooth)

- **Defending against alignment-breaking attacks via robustly aligned LLM**  
  **Conference**: ACL 2024  
  [Paper](https://arxiv.org/abs/2309.14348)

- **Selectively answering ambiguous questions**  
  **Conference**: EMNLP 2023  
  [Paper](https://arxiv.org/abs/2305.14613)

- **RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2403.13031) | [GitHub](https://github.com/eurekayuan/RigorLLM)

- **INSIDE: LLMs’ internal states retain the power of hallucination detection**  
  **Conference**: ICLR 2024  
  [Paper](https://arxiv.org/abs/2402.03744) | [GitHub](https://github.com/alibaba/eigenscore)

- **SmoothLLM: Defending large language models against jailbreaking attacks**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2310.03684) | [GitHub](https://github.com/arobey1/smooth-llm)
  
<h4 id="inference-prompting-based">Prompting-Based</h4>

- **Generating with confidence: Uncertainty quantification for black-box large language models**  
  **Conference**: TMLR 2024  
  [Paper](https://arxiv.org/abs/2305.19187) | [GitHub](https://github.com/zlin7/UQ-NLG)

- **The curious case of hallucinatory (un)answerability: Finding truths in the hidden states of overconfident large language models**  
  **Conference**: EMNLP 2023  
  [Paper](https://arxiv.org/abs/2310.11877) | [GitHub](https://github.com/lovodkin93/unanswerability)

- **Characterizing LLM abstention behavior in science QA with context perturbations**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2404.12452) | [GitHub](https://github.com/bbwen/llm_scienceqa)

- **Alignment for honesty**  
  **Conference**: NeurIPS 2024  
  [Paper](https://arxiv.org/abs/2312.07000) | [GitHub](https://github.com/GAIR-NLP/alignment-for-honesty)

- **Can AI assistants know what they don’t know?**  
  **Conference**: ICML 2024 Poster  
  [Paper](https://arxiv.org/pdf/2401.13275) | [GitHub](https://github.com/OpenMOSS/Say-I-Dont-Know)

- **Defending large language models against jailbreaking attacks through goal prioritization**  
  **Conference**: ACL 2024  
  [Paper](https://aclanthology.org/2024.acl-long.481/) | [GitHub](https://github.com/thu-coai/JailbreakDefense_GoalPriority)

- **Intention analysis makes LLMs a good jailbreak defender**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2401.06561) | [GitHub](https://github.com/alphadl/SafeLLM_with_IntentionAnalysis)

- **Prompt-driven LLM safeguarding via directed representation optimization**  
  **Conference**: ICML 2024  
  [Paper](https://arxiv.org/abs/2401.18018) | [GitHub](https://github.com/chujiezheng/LLM-Safeguard)

- **Robust prompt optimization for defending language models against jailbreaking attacks**  
  **Conference**: ICLR 2024 Workshop  
  [Paper](https://arxiv.org/abs/2401.17263)

- **Defending jailbreak prompts via in-context adversarial game**  
  **Conference**: EMNLP 2024  
  [Paper](https://arxiv.org/abs/2402.13148)

- **Self-evaluation improves selective generation in large language models**   
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2312.09300)

- **Fight back against jailbreaking via prompt adversarial tuning**  
  **Conference**: NeurIPS 2024  
  [Paper](https://arxiv.org/abs/2402.06255) | [GitHub](https://github.com/PKU-ML/PAT)

- **Bergeron: Combating adversarial attacks through a conscience-based alignment framework**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2312.00029) | [GitHub](https://github.com/matthew-pisano/Bergeron)

- **Jailbreak and guard aligned language models with only few in-context demonstrations**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2310.06387) | [GitHub](https://adv-icl.github.io/)

- **Defending ChatGPT against jailbreak attack via self-reminders**  
  **Conference**: Nature  
  [Paper](https://www.nature.com/articles/s42256-023-00765-8)

- **The art of defending: A systematic evaluation and analysis of LLM defense strategies on safety and over-defensiveness**  
  **Conference**: ACL 2024  
  [Paper](https://arxiv.org/abs/2401.00287)
---


<h3 id="inference-output-processing">Output-Processing</h3>
<h4 id="inference-self-evaluation">Self-Evaluation</h4>

- **LLM self defense: By self-examination, LLMs know they are being tricked**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2308.07308)

- **Language models (mostly) know what they know**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/pdf/2207.05221)

- **The art of defending: A systematic evaluation and analysis of LLM defense strategies on safety and over-defensiveness**  
  **Conference**: ACL 2024  
  [Paper](https://aclanthology.org/2024.findings-acl.776/)

- **Adaptation with self-evaluation to improve selective prediction in LLMs**  
  **Conference**: EMNLP 2023  
  [Paper](https://arxiv.org/abs/2310.11689)

- **Self-evaluation improves selective generation in large language models**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2312.09300)

- **I could’ve asked that: Reformulating unanswerable questions**  
  **Conference**: EMNLP 2024  
  [Paper](https://www.arxiv.org/abs/2407.17469) | [GitHub](https://github.com/wenting-zhao/couldask)
<h4 id="inference-llm-collaboration">LLM Collaboration</h4>



- **Reducing conversational agents’ overconfidence through linguistic calibration**  
  **Conference**: TACL 2022  
  [Paper](https://arxiv.org/abs/2012.14983)

- **SELF-GUARD: Empower the LLM to safeguard itself**   
  **Conference**: NAACL 2024  
  [Paper](https://arxiv.org/abs/2310.15851)

- **Bergeron: Combating adversarial attacks through a conscience-based alignment framework**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2312.00029) | [GitHub](https://github.com/matthew-pisano/Bergeron)

- **Jailbreaker in jail: Moving target defense for large language models**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2310.02417)

- **Don’t hallucinate, abstain: Identifying LLM knowledge gaps via Multi-LLM collaboration**  
  **Conference**: ACL 2024  
  [Paper](https://aclanthology.org/2024.acl-long.786.pdf) | [GitHub](https://github.com/BunsenFeng/AbstainQA)

- **Autodefense: Multi-agent LLM defense against jailbreak attacks**  
  **Conference**: N/A  
  [Paper](https://arxiv.org/abs/2403.04783) | **Project**: SafeGenAi


<h2 id="abstention—evaluation-benchmarks">Abstention Evaluation Benchmarks</h2>

<h3 id="query">Query</h3>

- **Know what you don’t know: Unanswerable questions for SQuAD**  
  **Conference**: ACL 2018  
  [Paper](https://arxiv.org/abs/1806.03822) | [Website](https://worksheets.codalab.org/worksheets/0x9a15a170809f4e2cb7940e1f256dee55/)  

- **Natural Questions: A benchmark for question answering research**  
  **Conference**: TACL 2019  
  [Paper](https://aclanthology.org/Q19-1026/) | [Website](https://ai.google.com/research/NaturalQuestions)  

- **MuSiQue: Multihop questions via single-hop question composition**  
  **Conference**: TACL 2022  
  [Paper](https://arxiv.org/pdf/2108.00573) | [GitHub](https://github.com/stonybrooknlp/musique)  

- **CoQA: A conversational question answering challenge**  
  **Conference**: TACL 2019  
  [Paper](https://arxiv.org/abs/1808.07042) | [Website](https://stanfordnlp.github.io/coqa/)  

- **QuAC: Question answering in context**  
  **Conference**: ACL 2018  
  [Paper](https://arxiv.org/abs/1808.07036) | [Website](https://quac.ai/)  

- **AmbigQA: Answering ambiguous open-domain questions**  
  **Conference**: EMNLP 2020  
  [Paper](https://arxiv.org/abs/2004.10645) | [Website](https://nlp.cs.washington.edu/ambigqa/)  

- **SituatedQA: Incorporating extra-linguistic contexts into QA**  
  **Conference**: EMNLP 2021  
  [Paper](https://arxiv.org/abs/2109.06157) | [Website](https://situatedqa.github.io/)  

- **Do large language models know what they don’t know?**  
  **Conference**: ACL 2023  
  [Paper](https://arxiv.org/abs/2305.18153) | [GitHub](https://github.com/yinzhangyue/SelfAware)  

- **Known Unknown Questions: A new dataset for large language models**  
  **Conference**: ACL 2024  
  [Paper](https://arxiv.org/abs/2305.13712) | [GitHub](https://github.com/amayuelas/knowledge-of-knowledge)  

- **PubMedQA: A dataset for biomedical research question answering**  
  **Conference**: EMNLP 2019  
  [Paper](https://arxiv.org/abs/1909.06146) | [Website](https://pubmedqa.github.io/)  

<h3 id="model">Model</h3>

- **RealTimeQA: What’s the answer right now?**  
  **Conference**: NeurIPS 2023  
  [Paper](https://arxiv.org/abs/2207.13332) | [Website](https://realtimeqa.github.io/)  

- **PUQA: A dataset of out-of-knowledge questions for scientific literature**  
  **Conference**: Nature  
  [Paper](https://arxiv.org/html/2407.09413) | [GitHub](https://github.com/google/spiqa)  

- **Don’t hallucinate, abstain: Identifying LLM knowledge gaps via Multi-LLM collaboration**  
  **Conference**: ACL 2024  
  [Paper](https://aclanthology.org/2024.acl-long.786.pdf) | [GitHub](https://github.com/BunsenFeng/AbstainQA)  

- **POPQA: A question answering dataset on long-tail entities**  
  **Conference**: ACL 2023  
  [Paper](https://arxiv.org/html/2405.06524v1) | [GitHub](https://github.com/AlexTMallen/adaptive-retrieval)  

- **Simple entity-centric questions challenge dense retrievers**  
  **Conference**: EMNLP 2021  
  [Paper](https://arxiv.org/abs/2109.08535) | [GitHub](https://github.com/princeton-nlp/EntityQuestions)  

<h3 id="human-values">Human Values</h3>

- **RealToxicityPrompts: Evaluating neural toxic degeneration in language models**  
  **Conference**: EMNLP 2020  
  [Paper](https://arxiv.org/abs/2009.11462) | [Website](https://toxicdegeneration.allenai.org/)  

- **ToxiGen: A large-scale machine-generated dataset for adversarial and implicit hate speech detection**  
  **Conference**: ACL 2022  
  [Paper](https://arxiv.org/abs/2203.09509) | [GitHub](https://github.com/microsoft/ToxiGen)  

- **Latent Hatred: A Benchmark for Understanding Implicit Hate Speech**  
  **Conference**: EMNLP 2021  
  [Paper](https://arxiv.org/abs/2109.05322) | [GitHub](https://github.com/SALT-NLP/implicit-hate)  

- **ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation**  
  **Conference**: EMNLP 2023  
  [Paper](https://arxiv.org/abs/2310.17389) | [Dataset](https://huggingface.co/datasets/lmsys/toxic-chat)  

- **Beavertails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset**  
  **Conference**: NeurIPS 2023  
  [Paper](https://arxiv.org/abs/2307.04657) | [Website](https://sites.google.com/view/pku-beavertails)  

- **CValues: Measuring the Values of Chinese Large Language Models from Safety to Responsibility**  
  [Paper](https://arxiv.org/abs/2307.09705) | [Dataset](https://www.modelscope.cn/datasets/damo/CValues-Comparison/summary)  

- **Xstest: A Test Suite for Identifying Exaggerated Safety Behaviors in Large Language Models**  
  **Conference**: NAACL 2024  
  [Paper](https://arxiv.org/abs/2308.01263) | [GitHub](https://github.com/paul-rottger/exaggerated-safety)  

- **LatentJailbreak: A Benchmark for Evaluating Text Safety and Output Robustness of Large Language Models**  
  [Paper](https://arxiv.org/abs/2307.08487) | [GitHub](https://github.com/qiuhuachuan/latent-jailbreak)  

- **Do Anything Now: Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models**  
  [Paper](https://arxiv.org/abs/2308.03825) | [GitHub](https://github.com/verazuo/jailbreak_llms)  

- **Do-Not-Answer: Evaluating Safeguards in LLMs**  
  **Conference**: EACL 2024  
  [Paper](https://arxiv.org/abs/2308.13387) | [GitHub](https://github.com/Libr-AI/do-not-answer)  

- **XSafety: A Multilingual Benchmark for Safety Refusal of Large Language Models**  
  [Paper](https://arxiv.org/abs/2310.00905) | [GitHub](https://github.com/Jarviswang94/Multilingual_safety_benchmark)  

- **SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models**  
  **Conference**: ACL 2024  
  [Paper](https://arxiv.org/abs/2402.05044) | [GitHub](https://github.com/OpenSafetyLab/SALAD-BENCH)  

- **SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal Behaviors**  
  [Paper](https://arxiv.org/abs/2406.14598) | [Website](https://sorry-bench.github.io/)  

- **WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs**  
  [Paper](https://arxiv.org/abs/2406.18495) | [GitHub](https://github.com/allenai/wildguard)


<h3 id="general">General</h3>

- **The art of saying no: Contextual noncompliance in language models**  
  **Conference**: NeurIPS 2024 Track Datasets and Benchmarks Poster  
  [Paper](https://www.arxiv.org/abs/2407.12043) | [GitHub](https://github.com/allenai/noncompliance)  


<!--## Citation
If you find this repository useful, please cite the following paper:
```
@article{wen2024know,
  title={Know Your Limits: A Survey of Abstention in Large Language Models},
  author={Wen, Bingbing and Yao, Jihan and Feng, Shangbin and Xu, Chenjun and Tsvetkov, Yulia and Howe, Bill and Wang, Lucy Lu},
  journal={arXiv preprint arXiv:2407.18418},
  year={2024}
}
```
