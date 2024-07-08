# language-modeling
**Language modeling notes from Andrej Karpathy videos**

While watching Andrej Karpathy's great [list of videos on language modeling](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&si=P6YmUo5Wn5A_95cj) I created the following notes. 

Andrej Karpathy created [Makemore](https://github.com/karpathy/makemore), a character-level language model takes one text file as input and generates more things, with the educational purpose of teaching a wide choice of language models from Bigrams all the way to a Transformer.

Please note that the modeling approaches at the character level (predicting a character using the previous characters) are similar at the word level (predicting a word using the previous words).

## Content:

1. Bigram models
    - Notes: [Bigram](https://github.com/danielsimon4/language-modeling/blob/main/Bigrams%20(1).ipynb)
    - Video: [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2)
2. MLP model 
    - Notes: [MLP](https://github.com/danielsimon4/language-modeling/blob/main/MLP%20(2).ipynb)
    - Video: [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3)