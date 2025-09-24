
# LLM面试知识


## 简介

本仓库为大模型面试相关知识，由本人参考网络资源整理，欢迎阅读！


## 在线阅读

在线阅读链接：[llm_interview](https://cycloneboy.github.io/llm_interview/)


## 注意：

相关答案为自己网络搜索和AI辅助撰写，若有不合理地方，请指出修正，谢谢！


## 目录

* [Home](/)
* [大模型基础知识](/大模型基础知识/)
    * [attention](/大模型基础知识/attention.md)
        * [1.Attention](/大模型基础知识/attention.md?id=_1attention-1)
        * [2.Transformer](/大模型基础知识/attention.md?id=_2transformer)
        * [3.BERT](/大模型基础知识/attention.md?id=_3bert)
        * [4.MHA & MQA & GQA](/大模型基础知识/attention.md?id=_4mha-amp-mqa-amp-gqa)
        * [5.Flash Attention](/大模型基础知识/attention.md?id=_5flash-attention )
        * [6.Transformer常见问题](/大模型基础知识/attention.md?id=_6transformer常见问题 )
        * [7. Sliding Window Attention (SWA)](/大模型基础知识/attention.md?id=_7-sliding-window-attention-swa )
        * [8. Paged Attention (vllm)](/大模型基础知识/attention.md?id=_8-paged-attention-vllm )
        * [9.多头潜在注意力机制 (MLA) （Deepseek v3）](/大模型基础知识/attention.md?id=_9多头潜在注意力机制-mla-（deepseek-v3） )
        * [10. 长上下文](/大模型基础知识/attention.md?id=_10-长上下文)
    * [positional_encoding](/大模型基础知识/positional_encoding.md)
        * [1.位置编码基础](/大模型基础知识/positional_encoding.md?id=_1位置编码基础)
        * [2.旋转位置编码 RoPE篇](/大模型基础知识/positional_encoding.md?id=_2旋转位置编码-rope篇)
        * [3.ALiBi (Attention with Linear Biases)篇](/大模型基础知识/positional_encoding.md?id=_3alibi-attention-with-linear-biases篇)
        * [4.长度外推问题篇](/大模型基础知识/positional_encoding.md?id=_4长度外推问题篇)
        * [5.Yarn](/大模型基础知识/positional_encoding.md?id=_5yarn)
        * [6. 长上下文](/大模型基础知识/positional_encoding.md?id=_6-长上下文)
    * [混合专家模型 (MoE)](/大模型基础知识/moe.md)
* [大模型微调](/大模型微调/)
    * [有监督微调](/大模型微调/有监督微调.md)
        * [1.预训练](/大模型微调/有监督微调.md?id=_1预训练)
        * [2. 微调](/大模型微调/有监督微调.md?id=_2-微调)
        * [3.prompting](/大模型微调/有监督微调.md?id=_3prompting)
        * [4. adapter-tuning](/大模型微调/有监督微调.md?id=_4-adapter-tuning)
        * [5.lora](/大模型微调/有监督微调.md?id=_5lora)
        * [6.总结](/大模型微调/有监督微调.md?id=_6总结)
    * [强化学习微调](/大模型微调/强化学习微调.md)
        * [1. 策略梯度（pg）](/大模型微调/强化学习微调.md?id=_1-策略梯度（pg）)
        * [2. 近端策略优化(ppo)](/大模型微调/强化学习微调.md?id=_2-近端策略优化ppo)
        * [3. 大模型RLHF：PPO原理与源码解读](/大模型微调/强化学习微调.md?id=_3-大模型rlhf：ppo原理与源码解读)
        * [4. DPO](/大模型微调/强化学习微调.md?id=_4-dpo)
        * [5. 相关问题](/大模型微调/强化学习微调.md?id=_5-相关问题)
        * [6. GRPO](/大模型微调/强化学习微调.md?id=_6-grpo)
        * [7. DAPO](/大模型微调/强化学习微调.md?id=_7-dapo)
        * [8. GSPO](/大模型微调/强化学习微调.md?id=_8-gspo)
        * [9. DCPO](/大模型微调/强化学习微调.md?id=_9-dcpo)
        * [10. 参考链接](/大模型微调/强化学习微调.md?id=_10-参考链接)
    * [PPO](/大模型微调/ppo.md)
* [分布式训练](/分布式训练/)
    * [并行训练](/分布式训练/并行训练.md)
        * [1.概述](/分布式训练/并行训练.md?id=_1概述)
        * [2.数据并行](/分布式训练/并行训练.md?id=_2数据并行)
        * [3.流水线并行](/分布式训练/并行训练.md?id=_3流水线并行)
        * [4.张量并行](/分布式训练/并行训练.md?id=_4张量并行)
        * [5.序列并行](/分布式训练/并行训练.md?id=_5序列并行)   
        * [6.多维度混合并行](/分布式训练/并行训练.md?id=_6多维度混合并行)
        * [7.自动并行](/分布式训练/并行训练.md?id=_7自动并行)
        * [8.moe并行](/分布式训练/并行训练.md?id=_8moe并行)
        * [9.总结](/分布式训练/并行训练.md?id=_9总结)
    * [deepspeed](/分布式训练/deepspeed.md)
* [强化学习](/强化学习/)
* [大模型推理](/大模型推理/)
* [大模型应用](/大模型应用/)
    * [检索增强RAG](/大模型应用/检索增强RAG.md)
    * [智能体Agent](/大模型应用/智能体Agent.md)
    * [Text2SQL](/大模型应用/Text2SQL.md)
* [大模型论文](/大模型论文/)
* [大模型面试题](/大模型面试题)
* [视频课程学习](/视频课程学习/)
    * [CS336: Language Modeling from Scratch](/视频课程学习/cs336_2025.md)
    * [CS231n: Deep Learning for Computer Vision](/视频课程学习/cs231n_2025.md)
    * [Large Language Model Agents](/视频课程学习/llm_agent_2024.md)
