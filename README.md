# moco_distill

The code repository for [The Single-Multi Evolution Loop for Self-Improving Model Collaboration Systems](./), based on the model collaboration algorithms in [MoCo](https://github.com/BunsenFeng/model_collaboration).

## Quick Start

```
git clone https://github.com/BunsenFeng/model_collaboration.git
cd model_collaboration
conda env create -f environment.yml
conda activate model_collaboration
pip install modelco
```

First, a quick test to see if you can successfully run model collaboration algorithms in MoCo.

```
python -m model_collaboration.main -c model_collaboration/test_config.json
```

There will be a log file in `model_collaboration/logs/`. If successful:

```
cd model_collaboration
git clone https://github.com/BunsenFeng/moco_distill.git
cd ..
python -m model_collaboration.moco_distill.dev_create
```

Then run the basic single-multi evolution loop with the BLEND dataset and [Sparta](https://arxiv.org/abs/2506.04721) models, with supervised KD or multi-student KD:

```
python -m model_collaboration.moco_distill.supervised_kd -c model_collaboration/moco_distill/config.json
```

```
python -m model_collaboration.moco_distill.multi_student_kd -c model_collaboration/moco_distill/config.json
```

There will be a log directory in `model_collaboration/moco_distill/logs/`. In `score_dict.json`, you will find the performance of single models and multi-model systems across iterations in lists. You can also find generated texts and detailed logs in `generation_logs/`.

## Config files
You can run the single-multi evolution loop with technically any model collaboration algorithm in MoCo. Visit [link](https://github.com/BunsenFeng/model_collaboration/blob/dev/docs/user_readme.md) to learn how to create config files for different model collaboration algorithms in MoCo, specifying models, datasets, hyperparameters, etc.

The only extra thing is the `"rounds": 3` hyperparameter in the config file (note the s in `rounds`), which indicates how many rounds of single-multi evolution loop to run. Default 3.

The settings we used in the paper:

`method`: `api_trained_router`, `text_multiagent_refine`, `logit_logit_fusion` (requires same architecture models), `weight_dare_ties` (requires same architecture models), and additionally `text_llm_blender`, `text_multiagent_finetuning`, `api_graph_routing`.

`(task, task_type)`: `(agieval, multiple_choice)`, `(arc_challenge, multiple_choice)`, `(mmlu_redux, multiple_choice)`, `(bbh, exact_match)`, `(gsm8k, general_verifier)`, `(math, general_verifier)`, `(gpqa_diamond, multiple_choice)`, `(tablemwp_multiple_choice, multiple_choice)`, `(theoremqa, general_verifier)`, `(wikidyk, f1_match)`, `(popqa, f1_match)`, `(blend, multiple_choice)`, `(truthfulqa, multiple_choice)`, `(sciriff, exact_match)`, `(human_interest, reward_model)`.

`model_names`: `["bunsenfeng/yuru_qw_wizardlm", "bunsenfeng/yuru_qw_sharegpt", "bunsenfeng/yuru_qw_oasst1"]` for setting 1 (same architecture experts), `["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]` for setting 2.

## Reference

If this repository is useful for your research, please consider citing:

```
@article{feng2026moco,
  title={MoCo: A One-Stop Shop for Model Collaboration Research},
  author={Feng, Shangbin and Bai, Yuyang and Yang, Ziyuan and Wang, Yike and Tan, Zhaoxuan and Yan, Jiajie and Lei, Zhenyu and Ding, Wenxuan and Shi, Weijia and Wang, Haojin and others},
  journal={arXiv preprint arXiv:2601.21257},
  year={2026}
}

@article{feng2026single,
  title={The Single-Multi Evolution Loop for Self-Improving Model Collaboration Systems},
  author={Feng, Shangbin and Panaganti, Kishan and Tsvetkov, Yulia and Yu, Wenhao},
  journal={arXiv preprint arXiv:2602.05182},
  year={2026}
}
```
