## Generative Adverserial Network

Pytorch implementation of [GAN_Papers](https://github.com/Kyushik/Generative-Model.git) repository.

This repo provides explanations with references for loss function used in different types of GAN which is a critical part of neural network for its progress.

Also it provides end to end explanation for entire code

Contents:

S.NO | Topic Name | Usage | Link
---  | ---------  | ----- | ----
1    |  GAN        |      |
2    |  DC GAN     |      |
3    | Vanilla GAN | Generate Handwritten Digits|
4    | Conditional GAN | Generate Specific Digits|
5    | Progressive GAN | HUman faces with progressive GAN|
6    | Artistic Style Transfer GAN |     |
7    | Couples GAN |    |
8    | Super Resolution GAN |   |
9    | Pix2Pix GAN |    |
10   | Cycle GAN   |    |
11   | Vid2Vid GAN |    |

- Warm_up - contains basic codes to caculate loss function in pytorch with two methods

    Loss function explanation -

- GAN - Generative Adverserial Networks ,

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
    - DC_GAN - Deep Convolutional Generative Adverserial Networks

        Same as GAN

    - Vanilla GAN - Generate Handwritten Digits
    - Conditional GAN - Generate Specific Digits
    - Progressive GAN - HUman faces with progressive GAN
    - Artistic Style Transfer GAN
    - Couples GAN
    - Super Resolution GAN
    - Pix2Pix GAN
    - Super
    - Cycle GAN
    - Vid2Vid GAN




Credits:

- [GAN_Notebooks](https://github.com/Kyushik/Generative-Model.git)
- [GAN_TF](https://github.com/hwalsuklee/tensorflow-generative-model-collections)
- [GAN_Pytorch](https://github.com/contributeToWorld/PyTorch-GAN)



