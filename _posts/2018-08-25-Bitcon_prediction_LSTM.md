---
layout: post
title:  Predict Bitcoin price with LSTM
summary: How to use recurrent neural networks to forecast cryptocurrencies price 
featured-img: btc_prediction_plot
redirect_to:
  - https://theaisummer.com/Bitcon_prediction_LSTM
---

# Predict Bitcoin price with LSTM

Bitcoin and cryptocurrencies are eating the world. Sure, they all have a huge
slump over the past few months but do not be mistaken. Cryptocurrencies are here
to stay, and they are expected to overturn and reach higher levels than before.
Just think that the total market capitalization of crypto coins at the moments
is 215 billion USD and that number was around 800 billion on January.

So, who wouldn’t want to predict the future prices of bitcoins? And be sure that
most of the big bank, hedge funds and trading companies use some kind of
sophisticated algorithm to do exactly that. A sophisticated algorithm? What kind
of algorithm you may ask?

We guess that the answer includes deep learning techniques among others, but we
couldn’t be certain as no one is willing to reveal their secrets. And why should
they?

But I am moving away from the purpose of today’s article. The goal is to use a
simple Neural Network and try to predict future prices of bitcoin for a short
period of time. I decide to use recurrent networks and especially LSTM’s as they
proven to work really well for regression problems. Recurrent networks are
nothing more than simple networks with a feedback loop. What I mean, is that
apart from the standard input, they also use the information from previous states 
to compute the error gradient. They learn, in other words, from their own
history.

LSTM’s are an extension of the classic recurrent networks, which address the
vanishing gradient problem (the gradient tends to zero as the error propagates
through many layers recursively). The long-short term memory cell uses an input,
a forget and an output gate. Those gates help the network learns what to save,
what to forget, what to remember, what to pay attention and what to output.
Pretty neat right? Remember that a gate is nothing more than a simple multilayer
perceptron, but a smart combination of them can provide amazing results.

![LSTM cell]({{"/assets/img/posts/lstm_cll.jpg" | absolute_url}})

Let’s dive in a little.

Each LSTM cell has its cell state (c) and has the ability to add or remove information to it. 

The forget gate decides what to remove from the cell state(f), while the input gate (i) decides which values it will update. 

The tanh layer creates a vector of new candidate values (c_hat), that could be added to the state.

The input gate and the new candidate states are combined to update the cell state. 

Finally, it has to decide what to output (h). That is the responsibility of the output gate(o), which in fact filters the cell state from the unnecessary info. The output will be the feedback for the next round of training.

I know that is quite complicate, so feel free to read this one more time. I will wait…


![LSTM equations]({{"/assets/img/posts/lstm_equations.jpg" | absolute_url}})



But enough with the chit-chat. Let’s write some code. We should start by
downloading historical bitcoin prices over the past year from
[here](https://www.blockchain.com/charts/market-price?timespan=all). The dataset
is quite simple as it contains only the date and the price.

```python
from google.colab import files
files.upload()
df = pd.read_csv("market-price.csv",header=None)
dates=df['Date']
df.drop(['Date'], 1, inplace=True)
df.head()
```
<div>

<table  class="dataframe">
  <thead>
    <tr >
      <th></th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4363.054450</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4360.513317</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4354.308333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4391.673517</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4607.985450</td>
    </tr>
  </tbody>
</table>
</div>




Before we build our model, we should do a little data preprocessing:

-   Get rid of Date column (we will use it only to visualize our result)

-   Split into train and test set

-   Rescale prices to (0,1)

Now its time for the LSTM. The philosophy behind our approach is that we
feed the neural network with one price at a time and it forecasts the price
at the next moment. The model will consist of one LSTM layer with 100 units
(units is the dimension of its output and we can tune that number) , a
Dropout layer to reduce overfitting and a Dense( Fully Connected) layer
which is responsible for the actual prediction.

As you can see, it is a very simple model which can be greatly enhanced by
adding more layers and more data attributes (from twitter feeds to market
cap and volume). And you should use those, if you are really serious about
predict cryptocurrencies prices. Here I am just trying to give you a
baseline to work on.

For the training process, we will use Adam optimization and the mean squared
error as loss function.

By the way, Adam (Adaptive moment estimation) optimization is an enhancement
of stochastic gradient descent, that adapts the learning rates based on the
average first moment and the average second moment. More on that
[here](https://www.coursera.org/lecture/deep-neural-network/adam-optimization-algorithm-w9VCZ)
(from deeplearning.ai course on Coursera).

```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units= 100,activation= 'tanh'.input_shape=(None, 1)))
model.add(tf.keras.layers.Dropout(rate= 0.2))
model.add(tf.keras.layers.Dense(units= 1,  activation= 'linear'))

model.compile(optimizer= 'adam', loss= 'mse')
```

After 100 epochs, the model is trained and can be used to predict future
prices for the next month.

```python
model.fit(x=x_train,y=y_train,batch_size= 1,epochs= 100,verbose=True );

# Epoch 100/100
# 164/164 [==============================] - 1s 4ms/step - loss: 0.0020

inputs = min_max_scaler.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_price = model.predict(inputs)

plt.plot(dates[len(df)-prediction_days:],test_set[:, 0], color='red', label='Real BTC Price')
plt.plot(dates[len(df)-prediction_days:],predicted_price[:, 0], color = 'blue', label = 'Predicted BTC Price')

```

![Predict results]({{"/assets/img/posts/btc_prediction_plot.jpg" | absolute_url}})


Wow… The results are even better than I expected. And we achieve that in the
simplest way possible. Imagine if we use more layers, more epoch, fine tune
the parameters and, most importantly, add many more kinds of input. To give
you a hint we can:

Use twitter feed or news and perform sentiment analysis to capture the
general public opinion about bitcoin

-   Include statistical and financial models in the process

-   Economy general indexes about the current market situation

And to discourage you, a little note that the above results might not be as
perfect as they seem. The reason why is that eventually the network learns
to predict a very close value to the previous one in terms of minimizing the
mean squared error. In general, historic data are not the best way to
predict a price as they are prone to such misunderstandings.

However, the potential is here and there is no doubt that they can actually
be used for those problems especially if the are combined with different
architectures such as convolutional networks.

Well that’s all folks. I hope to have helped you, even a bit, understanding
what LSTM's are and how you can use them.

Adios…
