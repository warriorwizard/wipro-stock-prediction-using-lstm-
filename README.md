
<body lang=EN-US link="#0563C1" vlink="#954F72" style='word-wrap:break-word'>

<div class=WordSection1>

<p class=papertitle>Wipro stock prediction using LSTM</p>


<div class=WordSection6>

<p class=Abstract><i>Abstract</i>—The aim is to predict the future value of the
financial stocks of WIPRO. The future prediction is done using previous data
and using that data a machine learning model is trained which analyses the
trend based on its learning outputs the result. This paper focuses on use of regression,
minmaxscaler and LSTM based Machine learning to predict stock values. </p>

<p class=Keywords>Keywords--Close, high, low, LSTM model, minmaxscaler, open,
regression, and volume.</p>

<h1 style='margin-left:0in;text-indent:0in'><span style='font:7.0pt "Times New Roman"'>
</span>I.<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp; </span>Introduction
<i> </i></h1>

<p class=MsoBodyText><span lang=EN-IN>Big companies or financial institutions
hires thousand of peoples to find pattern and predict stock price and make
profit. It is difficult for them to manage and evaluate with so many factors
for proper prediction and finding the trend. This work can easily be performed
using machine learning model and this model can save a huge amount of money and
can do much better prediction with greater accuracy and with keeping all
variable factors in mind. </span></p>

<p class=MsoBodyText><span lang=EN-IN>Keeping this in mind and finding how much
money making it can be. a lot of research is being done in this field and
people are constantly trying to improve the model by applying various model
training algorithm. In this advent we also have tried to implement and improve
the model by using LSTM (long short term memory).</span></p>

<p class=MsoBodyText><span lang=EN-IN>Now if we are talking about the accuracy
then there are many factors which affect the price of a stock of a company. Maybe
it is due to short term company’s policy or maybe it is because of some action
taken by some high-ranking officer or the possibilities or causes are endless
and it is really hard and difficult to keep all the parameters in mind while
solving the problem that is predicting the trend. </span></p>

<p class=MsoBodyText><span lang=EN-IN>Despite of all this factors one major
thing that contributes how model will perform is on what dataset the model is
trained .and selecting the proper dataset is very important. Sometime even if
people select the right dataset, then also, they take wrong time interval from
which they must train the model. For example, if you want to predict short term
then you should not train model with long term dataset. The model will skip
short term prediction and if model is trained on relatively short-term data,
then the model will have insufficient data to learn from and it will not
perform up to the mark as expected. </span></p>

<p class=MsoBodyText><span lang=EN-IN>So keeping all this factors in mind we
have picked Wipro stock market dataset from Yahoo Finance. The dataset consist
of various variables such as:  prev close, open, high, low, last, close, volume.
And each contain stock price of stock of various time. Volume contains
information of stock data and amount of stock traded from one owner to another.
</span></p>

<p class=MsoBodyText><span lang=EN-IN>The dataset is divided into two parts one
is train data and other is test data. </span><span lang=X-NONE>Regression and
LSTM models are engaged for this conjecture separately.</span><span lang=EN-IN>
Regression minimizes the loss and LSTM remembers the data for long term
prediction.</span></p>

<p class=MsoBodyText><span lang=EN-IN>&nbsp;</span></p>

<h1 style='text-indent:0in'><span style='font:7.0pt "Times New Roman"'>
</span>II.<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp; </span>R<span
style='font-size:8.0pt'>ELATED</span> W<span style='font-size:8.0pt'>ORK</span></h1>

<p class=MsoNormal>Hello </p>

<h1 style='margin-left:0in;text-indent:0in'><span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>III.<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp; </span>M<span
style='font-size:8.0pt'>ETHODOLOGY</span></h1>

<p class=MsoBodyText><span lang=EN-IN>It is difficult to predict the stock
valuation and stock price in future and with proper accuracy can be tricky work
to do, so we are using AI and machine learning model to learn from previous
dataset and predict the trend of the market or stock with better accuracy. For
training the model the dataset plays an important role, so for that reason we
have taken the data from Yahoo finance. The dataset comprises of stock value
from year 2000 to 2021 and contains about 5300 entries which consist of
previous closing rate, open, high, low, last, close volume, turnover, trades
and many essential data. We have converted it into data-frame using pandas. We
have split the dataset into train and test data set in ratio 80 to 20
respectively. </span></p>

<p class=MsoBodyText><span lang=EN-IN>Although machine learning has many
algorithms for prediction but we would like to stick with LSTM. We are using
LSTM because as its name suggest it remembers long as well as short term memory.</span></p>

<p class=MsoBodyText style='text-indent:0in'><span lang=X-NONE>&nbsp;</span></p>

<p class=MsoBodyText style='text-indent:0in'><i><span lang=EN-IN>A</span><span
lang=X-NONE>. Long Short Term Memory (LSTM) Network Based Model</span></i></p>

<p class=MsoBodyText><span lang=X-NONE><img border=0 width=335 height=749
id="Picture 1" src="stock_market_prediction%20(1)_files/image001.png"></span></p>

<p class=MsoBodyText align=center style='text-align:center'><span lang=X-NONE
style='font-size:8.0pt;line-height:95%'>Fig.</span><span lang=EN-IN
style='font-size:8.0pt;line-height:95%'>1</span><span lang=X-NONE
style='font-size:8.0pt;line-height:95%'> LSTM Layers</span></p>

<p class=MsoBodyText><span lang=X-NONE>LSTM is the advanced model of
Recurrent-Neural Networks (RNN) in which the information belonging to preceding</span><span
lang=EN-IN> state</span><span lang=X-NONE> persists. these are distinct from
RNNs as they contain long time dependencies and RNNs works on locating the
relationship between the latest and the</span><span lang=EN-IN> current</span><span
lang=X-NONE> information. This suggests that the </span><span lang=EN-IN>interval
of information </span><span lang=X-NONE>is quite smaller than that to LSTM. the
principle cause behind using this model in stock market prediction is that the
predictions depends on huge amounts of records and are generally depending on
the long-time records of the marketplace. So LSTM regulates errors by means of
giving an resource to the RNNs thru retaining records for older tiers making
the prediction more accurate. for this reason proving itself as plenty extra
reliable in comparison to different strategies. considering the fact that stock
market involves processing of massive </span><span lang=EN-IN>data</span><span
lang=X-NONE>, the gradients with respect to the weight matrix may additionally
turn out to be very small and can degrade the learning fee. </span><span
lang=EN-IN>The </span><span lang=X-NONE>corresponds to the problem of Vanishing
Gradient. LSTM prevents this from happening. The LSTM consists of a remembering
cell, input gate, output gate and a forget gate. The cell remembers the value
for long term propagation and the gates regulate them.</span></p>

<p class=MsoBodyText><span lang=EN-IN>In this paper we have made a sequential
model consisting of four layers of LSTM stacked over one another and a with the
output value as mentioned in Fig. 1. each and at last a dense layer is created
where each neuron is connected to every other neuron and it takes input of 50
and gives an output of 1. Additionally, we have also created dropout layer with
0.2 percent of dropout in every layer, this will prevent the model from over
fitting which is a major drawback while training the model. </span><span
lang=X-NONE>The</span><span lang=EN-IN> model is compiled with mean square cost
function to maintain the error and loss.</span></p>

<p class=MsoBodyText><span lang=EN-IN>We have taken 25 epochs with batch size
of 32 and for time series we have saved data by using callback method and
saving progress and using tensorboard to visualize it.</span></p>

<h1 style='margin-left:0in;text-indent:0in'><span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>IV.<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp; </span>Experiment
Result</h1>

<p class=MsoNormal style='text-align:justify;text-justify:inter-ideograph'>The
trained model is trained and tested as in proposed dataset.</p>

<p class=MsoNormal style='text-align:justify;text-justify:inter-ideograph'>We
have taken the dataset of Wipro Limited from yahoo finance and split it as it
was proposed.</p>

<p class=MsoNormal style='text-align:justify;text-justify:inter-ideograph'>&nbsp;</p>

<h2>A.<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp; </span>LSTM
based model result</h2>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><img border=0 width=343 height=245 id="Picture 2"
src="./stock_market_prediction%20(1)_files/image002.png" alt="lstm model result"></p>

<p class=MsoNormal><span style='font-size:8.0pt'>Fig. plot between actual and
predicted stock prices</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify;text-justify:inter-ideograph'>The
black line denotes the actual wipro stock prices in this timeframe while the
green line predicts the price of wipro stock predicted by the respective
trained model prescribed above. The distance between the two lines shows how
much efficient the LSTM model is. The model resulted the RMSE(root mean square
error) of 29.131674 when actual price vs predicted price is calculated. </p>

<p class=MsoNormal style='text-align:justify;text-justify:inter-ideograph'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify;text-justify:inter-ideograph'><img
border=0 width=335 height=226 id="Picture 4"
src="./stock_market_prediction%20(1)_files/image003.png"></p>

<p class=MsoNormal style='text-align:justify;text-justify:inter-ideograph'>&nbsp;</p>

<p class=MsoNormal><span style='font-size:8.0pt'>Fig 3. Epoch vs loss</span></p>

<p class=MsoNormal style='text-align:justify;text-justify:inter-ideograph'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify;text-justify:inter-ideograph'>The
above fig 3. Show epoch vs loss data. We have just taken 25 epochs because that
concluded us with very little loss in training thus providing high accuracy
than another model such as RNN.</p>

<p class=MsoNormal style='text-align:justify;text-justify:inter-ideograph'>&nbsp;</p>

<h1 style='margin-left:0in;text-indent:0in'><span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>V.<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp; </span>Conclusion</h1>

<p class=MsoNormal align=left style='text-align:left'>This paper was an attempt
to make model and try to predict the price with better accuracy and less RMSE
score. The model was a conclusion of the LSTM model that we have used in this paper.
The techniques have shown positive development in predicting future prices of
various stock prices of various companies. The LSTM model is quite good and
efficient.</p>

<p class=MsoNormal align=left style='text-align:left'>                Our
future goals are to make the model more accurate will try to tweak the model
with more parameters and will try to provide more dataset to get trained. Our
future goal is to make a model with higher accuracy and efficiency not only
this we will try to implement other factors which affect stock market like news
circulating about a company or organization and based on the sentiment
predicting what would be the output of it.</p>

<p class=MsoNormal align=left style='text-align:left'> </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>
</div>

<b><span lang=X-NONE style='font-size:10.0pt;font-family:"Times New Roman",serif;
color:red;letter-spacing:-.05pt'><br clear=all style='page-break-before:auto'>
</span></b>

<div class=WordSection7>

<p class=MsoNormal><span style='color:red'>&nbsp;</span></p>

</div>

</body>

</html>
