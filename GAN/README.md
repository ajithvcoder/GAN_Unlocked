### GAN - Generative Adverserial Networks

- Paper - https://arxiv.org/abs/1406.2661
- Abstract - We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.

    Loss function explanation -

    - [Notebook](GAN/GAN_notebook.ipynb)

    - Executing python file

            cd GAN/
            python gan.py

    - Loss function explanation -

        - loss function - Binary cross entropy loss

        Since its a real or fake problem author has used a binary cross entropy loss function

    - What is adverserial loss & Why ?

        The adversial loss is to tell the “distance” of two different distributions of P(G) and P(data), in other word, it is the JS divergence. If two distributions are nearly the same, JS divergence comes to a minimal value.


        ```adversarial_loss = torch.nn.BCELoss()```

        [torch_referece](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)

    - What is Generator loss & Why  ?

        - checking how much genrators image the discrimainator is able to classify as valid and then taking it as a loss and back propagating

        ```g_loss = adversarial_loss(discriminator(gen_imgs), valid)```

    - What is Discriminator loss & Why  ?

        - discriminators predictions for real images -> real loss
        - discriminators predictions for generated images -> generated/fake loss

         ```d_loss = (real_loss + fake_loss)/2```
