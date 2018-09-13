---
layout: post
title:  Decrypt Generative Artificial Intelligence and GANs
summary: What's the difference of generative and discriminative models and what is a GAN
featured-img: gan
---

# Decrypt Generative Artificial Intelligence and GANs

Hello all,

Today’s topic is a very exciting aspect of AI called generative artificial
intelligence. In a few words, generative AI refers to algorithms that make it
possible for machines to use things like text, audio files and images to
**create/generate** content. In a previous [post](https://sergioskar.github.io/Autoencoder/),
I talked about Variational Autoencoders and how they used to generate new images. 
I mentioned that they are a part of a bigger set of models called generative models and I will talk more
about them in a next article. So here we are.

As I briefly explained in that post, there are two types to modes.
Discriminative and generative. The first are the most common models, such as
convolutional or recurrent neural networks, which used to
distinguish/discriminate patterns in data in order to categorize them in class.
Application such as image recognition, skin-cancer diagnosis, Ethereum
prediction are all fall in the category of discriminative modes.

The latter are able to generate **new patterns** in data. As a result, they can
produce new images, new text, new music. To put it in a strict mathematical
form, discriminative models try to estimate the posterior probability p(y\|x),
which is the probability of an output sample (e.g the handwritten digit)
**given** an input sample (an image of a handwritten digit). On the other hand,
generative models estimate the joint probability p(x,y) , which is the
probability of both input sample and sample output to be true at the same time.
In reality, it tries to calculate **the distribution of a set of classes not the
boundary between them.**

Can you imagine the possibilities? Well you can take a glimpse of them by
looking at the current progress in the field and some existing applications.
Generative models have been used so far to produce [text from
images](https://arxiv.org/pdf/1711.10485.pdf), to [develop molecules in
oncology](http://www.oncotarget.com/index.php?journal=oncotarget&page=article&op=view&path%5B0%5D=14073&path%5B1%5D=44886),
to [discover new drugs](https://arxiv.org/abs/1708.08227) and to [transfer the
style](https://deepart.io/) of artists like Van Gogh to new images. And I pretty
sure you heard about Deepfakes, where they put celebrities faces on any sort of
video. And if you think you can tell the fakes apart from the real ones, forget
it. You can’t.

If you clicked on some of the above link, you may have noticed something that is
even more fascinating. All the applications have become possible due to
something called GANs. GANs or **Gererative Adversarial Networks** is the base
architecture behind most of generative applications. Of course, there are many
other cool models, such as Variational Autoencoders, Deep Boltzman machines,
Markov chains but GANs are the reason why there is so much hype in the last
three years around generative AI.

## What are Generative Adversarial Networks?

Generative Adversarial Networks were introduced in 2016 by Ian Goodfellow in one
of the most promising AI [paper](https://arxiv.org/pdf/1406.2661.pdf) in the
last decade. They are an unsupervised learning technique and they based on a
simple premise:

You want to generate new data. What do you do? You build two models**. You train
the first one to generate fake data and the second one to distinguish real from
fakes ones. And you put them compete against each other**.

Boom! There you have it. I wish it would be as simple as that. It isn’t. But
this is the main principle behind GANs.

Ok let’s get into some details. The first model is a neural network, called the
Generator. Generator’s job is to produce fake data with nothing but noise as
input. The second model, the Discriminator, receives as input both the real
images and the fake one (produced by the generator) and learns to identify if
the image is fake or not. As you put them contesting against each other and
train them simultaneously the magic begins:

The generator becomes better and better at image generation, as its ultimate
goal is to fool the discriminator. The discriminator becomes better and better
at distinguish fake from real images, as its goal is to not be fooled. The
result is that we now have incredibly realistic fake data from the
discriminator.

![GAN]({{"/assets/img/posts/gan.jpg" | absolute_url}})

The above image is a great analogy that describes the functionality between GAN.
The Generator can be seen as a forger who creates fraudulent documents and the
Discriminator as a Detective who tries to detect them. They participate in a
zero-sum game and they both become better and better as the time passes.

So far so good. We have the models and now we have to train them. Here is where
the problems begin to arise because it is not the standard method where we train
a neural network with gradient descent and a loss function. Here we have two
models competing against each other. So, what we do?

Well we are not sure. Optimization of GAN’s is one of the most active research
areas at the moment with many new papers appear constantly. I will try to
explain the base here and I am going to need some math and some game theory
(!!!) to do that. Please don’t leave. Stay with me and in the end, it is all
gonna make sense.

## How to train them?

We can consider that we have a [minimax](https://en.wikipedia.org/wiki/Minimax)
game here. To quote Wikipedia: “The maximin value of a player is the highest
value that the player can be sure to get without knowing the actions of the
other players; equivalently, it is the lowest value the other players can force
the player to receive when they know the player's action”

In other words, the first player tries to maximize his reward while minimizing
his opponent reward. The second player tries to accomplish the exact same goal.

In our case, **the Discriminator tries to maximize the probability of assigning
the correct label to both examples of real data and generated samples. While the
Generator tries to minimize the probability of the Discriminator’s correct
answer**.

We represent the loss as a minimax function:

![GAN]({{"/assets/img/posts/gan_training.jpg" | absolute_url}})

What do we have here?

The discriminator tries to maximize the function; therefore, we can perform
gradient ascent on the objective function. The generator tries to minimize the
function; therefore, we can perform gradient descent on the function. By
alternating between gradient ascent and descent, the models can be trained.

The training is stopped when the discriminator can’t maximize the function and
the generator can’t minimize it. In game theory terms, they reach Nash
equilibrium.

```python
def get_gan_network(discriminator, random_dim, generator, optimizer):
    gan_input = Input(shape=(random_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

def train(epochs=1, batch_size=128):
    # Get the training and testing data
    x_train, y_train, x_test, y_test = load_minst_data()
    batch_count = x_train.shape[0] / batch_size

    # Build our GAN netowrk
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, random_dim, generator, adam)

        for e in range(1, epochs+1):
                for _ in range(batch_count):
                    # Get a random set of input noise and images
                    noise = np.random.normal(0, 1, size=[batch_size, random_dim])
                    image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

                    # Generate fake images
                    generated_images = generator.predict(noise)
                    X = np.concatenate([image_batch, generated_images])

                    # Labels for generated and real data
                    y_dis = np.zeros(2*batch_size)
                    # One-sided label smoothing
                    y_dis[:batch_size] = 0.9

                    # Train discriminator
                    discriminator.train_on_batch(X, y_dis)

                    # Train generator
                    noise = np.random.normal(0, 1, size=[batch_size, random_dim])
                    y_gen = np.ones(batch_size)
                    gan.train_on_batch(noise, y_gen)
```

I hope you’re still here. This is the main idea and is called adversarial
training. Of course, there are several pitfalls which occur frequently such as:

-   The model parameters oscillate and never converge,

-   The discriminator gets too successful that the generator gradient vanishes

-   It’s highly sensitive to the hyperparameter

-   The generator produces limited varieties of samples

Over the past few years, there is a big contribution from scientists to solve
these problems and we can say that a lot of progress has been made. Just do a
quick search on [arxiv-sanity](http://www.arxiv-sanity.com/). It’s still very
early, though. Remember. GAN’s exists for less than three years.

I will close with some key facts. If you skipped the whole article it’s ok. But
don’t skip those:

-   Generative artificial intelligence is used to generate new data from real
    ones

-   The most prominent model of GAI is Generative Adversarial network.

-   GAN’s are two neural networks participated in a game. The first tries to
    produce new fake data and the second tries to tell them apart from real
    ones. As they trained, they both get better at what they do.

-   There is work that needs to be done on GAN’s training

-   Real time applications of GAI are … (how can I describe it in a word?
    Hmmm….) HUUUUUGE.


Finito…
