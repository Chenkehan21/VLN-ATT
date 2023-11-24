# Everyday Object Disrupts Vision-and-Language Navigation Agent via Backdoor

This repository is the official implementation of "Everyday Object Disrupts Vision-and-Language Navigation Agent via Backdoor"

Vision-and-Language Navigation (VLN) requires an agent to dynamically explore environments following natural language. The VLN agent, closely integrated into our daily lives, poses a substantial threat to the security of privacy and property upon the occurrence of malicious behavior. However, this serious issue has long been overlooked. In this paper, we pioneer the exploration of an object-aware backdoored VLN, achieved by implanting object-aware backdoors during the training phase. Tailored to the unique VLNnature of cross-modality and continuous decision-making, we propose a novel backdoored VLN paradigm: IPR Backdoor. This enables the agent to act in abnormal behavior once encountering the object triggers during language guided navigation in unseen environments, thereby executing an attack on the target scene. Our experiments demonstrate the effectiveness of our method in both physical and digital space across different VLN agents, as well as its robustness to various visual and textual variations. Additionally, our method also well ensures the navigation performance in normal scenarios with remarkable stealthiness.


![framework](framework.png "Framework")


## Installation

1. Install Matterport3D Simulator. Please follow the instructions [here](https://github.com/peteanderson80/Matterport3DSimulator).
2. Install requirements:

```bash
conda create --name vlnatt python=3.9
conda activate vlnhamt
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install -r requirement.txt
```

3. Download data from Dropbox and files and folds should be organized as follows:

   ```bash
   VLN-ATT
   |--datasets
      |--
   ```
