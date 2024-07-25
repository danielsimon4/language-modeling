# language-modeling
### Language modeling notes from Andrej Karpathy videos

I took the following notes from Andrej Karpathy's great video lecture series [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html). **Please refer to his video lectures while reading these notebooks.**

<br>

## Makemore:

Andrej Karpathy created [Makemore](https://github.com/karpathy/makemore), an autoregressive **character-level language model** that takes one text file as input, and generates more things like it. In his videos, he teaches how to build it using Bigram, MLP, and WaveNet architectures. We will train Makemore with the [names.txt](https://github.com/danielsimon4/language-modeling/blob/main/names.txt) dataset from [ssa.gov](https://www.ssa.gov/oact/babynames/), that contains the most common 32K names for the year 2018, and Makemore will generate more names.

1. **Bigram models**
    - Notebook: [Bigram](https://github.com/danielsimon4/language-modeling/blob/main/Bigram.ipynb)
    - Video: [Karpathy, Andrej. (2022). YouTube. *The spelled-out intro to language modeling: building makemore*](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2)
2. **MLP model** 
    - Notebook: [MLP](https://github.com/danielsimon4/language-modeling/blob/main/MLP.ipynb)
    - Video: [Karpathy, Andrej. (2022). YouTube. *Building makemore Part 2: MLP*](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3)
    - Reference: [Bengio et al. (2003). *A neural probabilistic language model*](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
3. **Activations and Gradients**
    - Notebook: [ActivationsAndGradients](https://github.com/danielsimon4/language-modeling/blob/main/ActivationsAndGradients.ipynb)
    - Video: [Karpathy, Andrej. (2022). YouTube. *Building makemore Part 3: Activations & Gradients, BatchNorm*](https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=5)
    - Reference:
        - [He et al. (2015). *Delving Deep into Rectifiers*](https://arxiv.org/pdf/1502.01852)
        - [Ioffe and Szegedy. (2015). *Batch normalization*](https://arxiv.org/pdf/1502.03167)
4. **Backpropagation**
    - Notebook: [Backpropagation](https://github.com/danielsimon4/language-modeling/blob/main/Backpropagation.ipynb)
    - Video: [Karpathy, Andrej. (2022). YouTube. *Building makemore Part 4: Becoming a Backprop Ninja*](https://www.youtube.com/watch?v=q8SA3rM6ckI&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=5)
5. **WaveNet**
    - Notebook: [WaveNet](https://github.com/danielsimon4/language-modeling/blob/main/WaveNet.ipynb)
    - Video: [Karpathy, Andrej. (2022). YouTube. *Building makemore Part 5: Building a WaveNet*](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=6)
    - Reference: [DeepMind. (2016). *WaveNet: A generative model for raw audio*](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

<br>

## NanoGPT:

Andrej Karpathy also created [NanoGPT](https://github.com/karpathy/nanoGPT?tab=readme-ov-file), a repository for training/finetuning medium-sized GPTs. In his videos, he teaches how to build a **Generatively Pretrained Transformer (GPT)**. We will train the transformer on a [Shakespere text](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset and genearte Shakespire like text.