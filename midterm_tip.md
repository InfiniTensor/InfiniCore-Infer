# 启元人工智能大赛中期报告

<!--
为了了解大家开发进展，遇到的问题并对赛事安排进行及时调整，增加一个非强制的中期报告。提交报告可以便于主办方进行调整，并且提交报告的队伍可以获取到官方**答疑/辅导**，帮助大家顺利完成比赛。

报告提交方式：
报告提交将在 大赛官网 -> 个人中心 -> 人工智能大赛 -> 作业提交 进行提交。提交内容与最终题目提交方式相同，采用PR提交的方式。
提交PR可以根据提交记录更好理清作品时间线，做好溯源，防止抄袭。

算子开发赛道和推理引擎赛道请分开提交。
队长提交。
-->

## 1. 参与赛道

<!-- 列出报名赛道、赛题；~~~~引擎开发赛道请标明所使用引擎，以及所对应的模型。赛道、赛题已经 -->
**九源统一智能计算架构算子开发**

参与了困难算子赛题T1-3-2 MLA 实现当中 tOPK 已经实现 分为普通的softmax 版本和deepseek 的sigmoid版本 并构建了测试程序 且通过 只做了nv的 考虑到沐曦可以转译cuda 后续的开发工作应该相对简单



**九源大模型推理引擎开发**
<!--
模型适配赛题
推理系统优化赛题
量化模型推理赛题
-->
例：
参与了模型适配赛题T2-1-1 选用InfiniCore-infer框架，适配tinymix模型 
后续打算继续适配 mistral 和deepseek 671b（T2-2-3） 

## 2. 技术路线（简要技术点）
使用基础的MoE推理过程 构建forward 函数 目前还未实现attention 所以用的是Gemm拼装计算的attention
在infinicore中自建算子Moe_dispatch 和 moe_combine 用以支持 moe模型的推理需要
<!-- 只列简要技术点，不写具体实现，用一句话概括 -->
考虑到 MoE 的共通性 可以利用deepseek开源的deepGemm 和FLASH MLA 以及DeepEP 嵌入提升性能

## 3. 时间线
7月21日-7月29日 调研 + 开发


## 4. 潜在风险 / 需协助

hot_wind@gpua18:~/InfiniCore-Infer/scripts$ srun --cpus-per-task=16 --mem=256G --export=ALL,INFINI_ROOT=/home/hot_wind/.infini,SAFETENSORS_MAX_HEADER_SIZE=1000000000 python tinymix.py --nvidia /home/shared/models/tinymix-8x1b-chat/
Loading model weights to host...
Traceback (most recent call last):
  File "/home/hot_wind/InfiniCore-Infer/scripts/tinymix.py", line 456, in <module>
    test()
  File "/home/hot_wind/InfiniCore-Infer/scripts/tinymix.py", line 450, in test
    model = TinyMixForCauslLM(model_path, device_type, ndev)
  File "/home/hot_wind/InfiniCore-Infer/scripts/tinymix.py", line 325, in __init__
    with safetensors.safe_open(file, "pt") as data:
safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge
srun: error: gpua18: task 0: Exited with exit code 1

在 测试 tinymix 的推理程序的时候 根据网上搜索的信息 这个要么是我的代码逻辑存在问题 要么是模型权重 需要重新下载 
我觉得后者的可能性较大

<!-- 完 -->