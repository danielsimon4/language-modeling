# Neural Networks & Transformers: A Deep Dive into Language Modeling

This [**online book**]() is inspired by Andrej Karpathy's great video lecture series, [***Neural Networks: Zero To Hero***](https://karpathy.ai/zero-to-hero.html). As I followed along with the series, I took comprehensive notes and, at times, simplified the code to make it more approachable. I have also added explanations before every piece of code to clarify key concepts and to ensure the material is accessible to readers with all levels of expertise.

In addition to the concepts covered in the videos, this book includes topics and even chapters on areas not discussed in the series, such as Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), providing a broader foundation in neural network architectures.

By the end of this book, you will be familiar with core concepts in neural networks, ranging from training simple models to implementing advanced architectures. Whether you are a beginner looking to get started or someone seeking deeper insights, I hope this resource aids you on your journey through the fascinating world of deep learning.

<br>

## Contents:

**Chapter 1: Introduction to Neural Networks**
- 1.1 

**Chapter 2: Bigram models**

**Chapter 3: MLP model**

**Chapter 4: Activations and Gradients**

**Chapter 5: Backpropagation**

**Chapter 6: WaveNet**

[Chapter 7: Generatively Pretrained Transformers (GPTs)](https://github.com/danielsimon4/language-modeling/blob/main/NanoGPT/GPT.ipynb)
- [7.1 Transformers](https://github.com/danielsimon4/language-modeling/blob/main/NanoGPT/GPT.ipynb#71-transformers)
- 7.2 Load dataset
- [7.3 Vocabulary size](https://github.com/danielsimon4/language-modeling/blob/main/NanoGPT/GPT.ipynb#73-vocabulary-size)
- 7.4 Tokenizer
- 7.5 Build dataset
- 7.6 Chunk, block size, and context
- 7.8 CUDA

**Chapter 8: GPT-2**






## Neural Networks:

Andrej Karpathy created [Makemore](https://github.com/karpathy/makemore), an autoregressive **character-level language model** that takes a text file with words as input, and generates more words like it. In his videos, he teaches how to build it using Bigram, MLP, and WaveNet architectures. We will train Makemore with the [names](https://github.com/danielsimon4/language-modeling/blob/main/Makemore/names.txt) dataset from [ssa.gov](https://www.ssa.gov/oact/babynames/), that contains the most common 32K names for the year 2018, and Makemore will generate more names.

1. **Bigram models**
    - Notebook: [Bigram](https://github.com/danielsimon4/language-modeling/blob/main/Makemore/Bigram.ipynb)
    - Video: [Karpathy, A. (2022). *The spelled-out intro to language modeling: building makemore*](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2)
2. **MLP model** 
    - Notebook: [MLP](https://github.com/danielsimon4/language-modeling/blob/main/Makemore/MLP.ipynb)
    - Video: [Karpathy, A. (2022). *Building makemore Part 2: MLP*](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3)
    - Reference: [Bengio et al. (2003). *A neural probabilistic language model*](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
3. **Activations and Gradients**
    - Notebook: [ActivationsAndGradients](https://github.com/danielsimon4/language-modeling/blob/main/Makemore/ActivationsAndGradients.ipynb)
    - Video: [Karpathy, A. (2022). *Building makemore Part 3: Activations & Gradients, BatchNorm*](https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4)
    - Reference:
        - [He et al. (2015). *Delving Deep into Rectifiers*](https://arxiv.org/abs/1502.01852)
        - [Ioffe & Szegedy. (2015). *Batch normalization*](https://arxiv.org/abs/1502.03167)
4. **Backpropagation**
    - Notebook: [Backpropagation](https://github.com/danielsimon4/language-modeling/blob/main/Makemore/Backpropagation.ipynb)
    - Video: [Karpathy, A. (2022). *Building makemore Part 4: Becoming a Backprop Ninja*](https://www.youtube.com/watch?v=q8SA3rM6ckI&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=5)
5. **WaveNet**
    - Notebook: [WaveNet](https://github.com/danielsimon4/language-modeling/blob/main/Makemore/WaveNet.ipynb)
    - Video: [Karpathy, A. (2022). *Building makemore Part 5: Building a WaveNet*](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=6)
    - Reference: [DeepMind. (2016). *WaveNet: A generative model for raw audio*](https://arxiv.org/abs/1609.03499)

<br>

## Transformers:

In his videos, Andrej Karpathy also teaches how to build a **Generatively Pretrained Transformer (GPT)** and how to replicate GPT-2. We are going to   

1. **GPT**
    - Notebook: [GPT](https://github.com/danielsimon4/language-modeling/blob/main/NanoGPT/GPT.ipynb)
    - Code: [GPT](https://github.com/danielsimon4/language-modeling/blob/main/NanoGPT/GPT.py)
    - Video: [Karpathy, A. (2023). *Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)
    - Reference:
        - [Vaswani et al. (2017). *Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
        - [Ba et al. (2016). *Layer Normalization*](https://arxiv.org/abs/1607.06450)
        - [He et al. (2015). *Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385)
        - [Srivastava et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*](https://jmlr.org/papers/v15/srivastava14a.html)
2. **GPT-2**
    - Notebook: [GPT2](https://github.com/danielsimon4/language-modeling/blob/main/NanoGPT/GPT2.ipynb)
    - Code: [GPT2](https://github.com/danielsimon4/language-modeling/blob/main/NanoGPT/GPT2.py)
    - Video: [Karpathy, A. (2023). *Let's reproduce GPT-2 (124M)*](https://www.youtube.com/watch?v=l8pRSuU81PU&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=10)
    - Reference:
        - [Vaswani et al. (2017). *Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
        - [Radford et al. (2019). *Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
        - [Brown et al. (2020). *Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165)