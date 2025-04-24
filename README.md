# GeneralModel
Making our own model

Timeline:

 - 2/11/25: started buidling this
 - 2/13/25: got basic transformer
 - 2/14/25: scalable barebones codebase 

## Run instructions

 - clone the repo
 - get in the source dir
 - run `data.py` in the data dir
 - run `train.py`
 - run `sampling.py`
 - enjoy model outputs

## Stuff to try

 - NanoGPT Infra

   - Get better data loading, text file sharding, bin making, etc, copy some from the Ngpt repo, talk to snow about efficient tokenization, etc 

   - combine the head and MHA classes into one single class

   - implement rope

   - cover any differences between my code and the deepseek or LLama 3 code, like get the stuff from the lillian weng transformers 2.0 

     - atp just make local attn layers from Noam lmao, and perhaps MLA

 - Pretraining
   - Data
     - Work on this
     - get annas archive data maybe
     - def getting Data from HF

   - Architecture

     - low memory, hyper efficient MoE 

        - https://arxiv.org/pdf/1907.05242
        - https://arxiv.org/pdf/2411.12364
        - https://arxiv.org/pdf/2407.04153

     - https://lunar-joke-35b.notion.site/Deepseek-v3-101-169ba4b6a3fa8090a7aacaee1a1cefaa
     - https://trite-song-d6a.notion.site/Deepseek-R1-for-Everyone-1860af77bef3806c9db5e5c2a256577d
     - https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/
     - https://ai.meta.com/research/publications/memory-layers-at-scale/
     - https://ai.meta.com/research/publications/explore-theory-of-mind-program-guided-adversarial-data-generation-for-theory-of-mind-reasoning/
     - https://arxiv.org/pdf/2403.09629
     - https://arxiv.org/abs/2405.16039
     - https://arxiv.org/pdf/2412.06769v1
     - https://sweet-hall-e72.notion.site/Why-are-Modern-Neural-Nets-the-way-they-are-And-Hidden-Hypernetworks-6c7195709e7b4abbada921875a951c54
     - Try to add some form of neural net based episodic memory/constant learning so that it can learn new stuff/use other examples to get better at new examples

   - Training Tricks
     - https://arxiv.org/pdf/2412.04318
     - https://arxiv.org/pdf/2405.20233
     - https://app.primeintellect.ai/speedrun/nanogpt
     - https://arxiv.org/pdf/2410.17897
     - https://arxiv.org/pdf/2411.19943
     - https://arxiv.org/pdf/2401.16380
     - https://arxiv.org/pdf/2404.07965
 
 - Post-training
   - https://arxiv.org/pdf/2410.12119
   - https://arxiv.org/pdf/2410.14251
   - https://arxiv.org/pdf/2411.15124
   - https://arxiv.org/abs/2501.07301
   - https://arxiv.org/pdf/2401.10020
   - https://arxiv.org/pdf/2402.03620
   - https://arxiv.org/pdf/2401.01286
 
 - RL
   - https://arxiv.org/pdf/2104.03113
   - https://arxiv.org/pdf/2412.07961
   - https://arxiv.org/pdf/2411.16905
   - https://arxiv.org/pdf/2410.02884
   - https://arxiv.org/pdf/2406.07394
   - https://arxiv.org/pdf/2402.05808
   - https://arxiv.org/pdf/2411.04282
   - https://arxiv.org/pdf/2410.18982
   - https://arxiv.org/pdf/2411.16489
   - https://arxiv.org/pdf/2410.01792
   - https://arxiv.org/pdf/2402.12875
   - https://arxiv.org/pdf/2404.00859
   - https://curvy-check-498.notion.site/Process-Reinforcement-through-Implicit-Rewards-15f4fcb9c42180f1b498cc9b2eaf896f

 - Curriculum learning:
   - https://arxiv.org/pdf/2310.09518
   - https://x.com/kalomaze/status/1876098342637519206

 - Sampling:
   - https://colab.research.google.com/drive/18-2Z4TMua-nwgCpIZo0lsKL6RDxH5Bvo?usp=sharing

OTHER:
 - benchmark with tinystories, or maybe some custom dataset of mine
 - Try things from charecter.ai, including:
 - MQA, multi-query attention, reduces KV cache size 8x
 - Hybrid attention, or interchanging local and global attention, they use a global attention 1/6 layers, and local attention set to 1024 ctx (comes from https://arxiv.org/abs/2004.05150v2)
 - KV caches are shared between different layers, like all the global neighbor layers share them, and the local neighbors share them (comes from https://arxiv.org/abs/2405.12981)
 - Train the whole thing in fp8, use the deepseek kernels 
 - Maximum @kaiokendev sparsity, butterfly matrices, high dropout, etc

NanoGPT baseline:
 - add newer transformers papers benchmarks
 - more Prod/speed features, maybe LLM.C stuff
 - better inference code
 - support for quantization and stuff
 - diff optimizers, diff learning rate schedulers, diff samplers, etc, switchable around
 - better data loader, more versatile
 - LLM mech-interp/visualization features, activation patches, etc
 - logging locally to matplotlib support

 - Tools and agent stuff
   - TBD, but try https://cohere.com/north
   - https://arxiv.org/pdf/2401.08500
   - https://arxiv.org/pdf/2402.05120
   - https://arxiv.org/pdf/2407.03502
   - https://arxiv.org/pdf/2408.06292
   - https://sakana.ai/namm/
   - https://sakana.ai/transformer-squared/
   - https://www.lindy.ai/ check out its flows
   - https://arxiv.org/pdf/2312.10997
   - Make an infinite context model/episodic memory model

 - Scale an API
   - learn system design ig

