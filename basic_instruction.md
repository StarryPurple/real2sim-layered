# 课题7：单目视频生成动态4D交互的 Real2Sim Agent

## 一、背景介绍

现实世界中，物体之间存在大量复杂的动态交互，例如抓取、碰撞、插入、折叠、拉伸等。这些交互不仅涉及物体的三维几何，还包含随时间变化的运动、形变、接触关系和物理约束。如何从普通视频中恢复这类动态交互，是 4D Vision 和 Embodied AI 中的重要问题。

相比静态 3D 数据，目前高质量的真实 4D 交互资产仍然十分缺乏。传统方法通常依赖多视角相机、深度传感器或人工建模，数据获取和制作成本较高。

近期 HAT\-4D、GPT\-6 \+ Blender、BVB 等工作展现了利用多模态大模型理解视频，并通过 Agent 调用 Blender、3D 重建和 4D 生成工具进行场景重建的潜力。其中 BVB 进一步表明，可以通过“视频理解 → Blender 程序化重建”的方式检验 Agent 对空间、时间和动态事件的理解能力。本课题以单目视频重建 4D 资产为切入点，使用 **MVOIK\-4D** 数据集作为 benchmark，探索如何搭建有效的 4D Agent Harness，实现从真实视频到动态 4D 交互资产的自动化构建。

关键词：4D Vision, Real2Sim, Agent, Tool Use, 3D Reconstruction

## 二、任务目标

1. 参考 HAT\-4D、BVB、GPT\-6 \+ Blender 等现有 3D / 4D Agent 工作，测试其在 **MVOIK\-4D** 数据集上的表现；

2. 分析 Agent 在重建 4D 交互过程中的难点，如物体形态、遮挡、形变、多物体关系和语义变化等；

3. 设计更好的 Agent Harness 或引入新的 3D / 4D Tool，提高 4D 资产重建的质量或效率。

## 三、可能的改进思路

以下为可尝试的方向（任选其一或多项结合）：

### 1\. 完善交互知识图谱

参考 HAT\-4D，对视频中的物体、动作、接触关系和时序变化进行结构化建模，利用 Interaction Knowledge Graph 帮助 Agent 更好地理解复杂多物体交互。

### 2\. 引入更好的 3D / 4D Tool

引入更强的 3D 重建、4D 重建或生成模型作为 Agent 可调用的工具，并根据不同场景和物体类型动态选择合适的工具。

### 3\. Physics\-aware Real2Sim

在 Real2Sim 过程中加入 Blender 等物理仿真工具，通过碰撞、接触和稳定性等物理约束发现并修正重建中的错误，提高交互过程的真实性。

## 四、参考论文与代码

### HAT\-4D

**HAT\-4D: Lifting Monocular Video for 4D Multi\-Object Interactions via Human\-Agent Collaboration**

Project: https://lijiaxin0111\.github\.io/HAT4D/
Code: https://github\.com/Lijiaxin0111/HAT\-4D\-Lifting\-Monocular\-Video\-for\-4D\-Multi\-Object\-Interactions\-via\-Human\-Agent\-Collaboration
Dataset: https://huggingface\.co/datasets/Lijiaxin0111/MVOIK\-4D

### BVB

**BVB: Benchmarking Agentic Video Understanding via Programmatic Reconstruction in Blender**

Paper: https://arxiv\.org/abs/2609\.15478

该工作让多模态 Agent 观察真实视频，并在统一的 Mini\-BVB Harness 中通过 Python / Blender 构建可编辑的动态场景，适合作为本课题中 **Video → Agent → Blender** 路线的重要参考。BVB 同时使用时空语义保持和视觉相似度评价 Agent 的重建结果。

### AHa\-3D / Real2Sim

Project: https://kevinxu02\.github\.io/real2sim\-indoor\-site/
Code: https://github\.com/KevinXu02/aha\-3d

### 其他 4D Tool

**L4GM: Large 4D Gaussian Reconstruction Model**
https://github\.com/nv\-tlabs/L4GM\-official

**DreamScene4D: Dynamic Multi\-Object Scene Generation from Monocular Videos**
https://github\.com/dreamscene4d/dreamscene4d

