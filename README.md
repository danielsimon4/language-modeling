# language-modeling
**Language modeling notes from Andrej Karpathy videos**

I took the following notes from Andrej Karpathy's great [list of videos on language modeling](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&si=P6YmUo5Wn5A_95cj). Please refer to his videos while reading these notebooks.

Andrej Karpathy created [Makemore](https://github.com/karpathy/makemore), a **character-level language model** that takes one text file as input and generates more things, with the educational purpose of teaching a wide choice of language models from Bigrams to Transformers. Please note that the modeling approaches at the character level (predicting a character using the previous characters) are similar at the word level (predicting a word using the previous words).

## Content:

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
    - Reference: