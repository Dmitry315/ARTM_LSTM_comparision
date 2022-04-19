<h1> ARTM_LSTM_comparision </h1>
Compare ARTM + RandomForestClassifier and LSTM neural network on emotion detection dataset

<h2> Dataset </h2>
Dataset from Kaggle: <br>
https://www.kaggle.com/datasets/ishantjuyal/emotions-in-text <br>
Consists of 1 csv file: "Emotion_final.csv" <br>
Columns: <br>
- Text: one sentence, that expresses emotion. <br>
- Emotion: target. Possible values: {"happy", "sadness", "anger", "fear", "love"} <br>
The "Text" column is quite clean, but need preprocessing <br>
<h2> Preprocessing </h2>
(Notebook was run on google collab, so for running on local PC you need to change file pathes in code)
1) Tokenization <br>
2) Lemmatization with pymorphy2 <br>
3) Remove stopwords. (nltk.corpus.stopwords.words('english')) <br>
4) Remove other useless words such as "feel", "feeling". Remove empty rows <br>
5) Do train test split (test_size = 0.2 * size) <br>
<h2> ARTM approach </h2>
Fit ARTM model with <n_topics> topics and get phi matrix. <br>
For each sentence get phi rows by words in sentences, and calculate exp(sum{word}{log(1 + phi[word] * n_topics)}). And also normalize features.<br>
(this works faster then calculate likelihood and have nearly the same accuracy in final result) <br>
And then aggregate this <n_topics> features with RandomForestClassifier from sklearn. <br>
Result is not so good: accuracy on test set ~ 0.41 <br>
(Optimal parameters: n_topics=200, decor_tau=50000.0, sparse_tau=100)<br>
<h2> LSTM approach</h2>
I don't know whether it is the best option for emotion detection, but I'v tried :). <br>
NN has the folowing structure: <br>
Sequential([ <br>
&emsp;  Embedding()<br>
&emsp;  LSTM(32)<br>
&emsp;  FullyConnected(out_dim = 64, 'relu')<br>
&emsp;  FullyConnected(aut_dim = 4, 'softmax')<br>
]) <br>
Optimizer - Adam, criteria - Categorical crossentropy, learing rate = 0.001 <br>
In notebook I implement it with tensorflow (in pytorch it appears more difficult, then I thought). <br>
Results are quite better: accuracy on test set ~ 0.75 <br>
This result get without any hyperparameter optimisation, unlike in ARTM try. <br>
<h2>Conclusions</h2>
Here we can see that ARTM is not sufficient for emotion detection problem and loses even to LSTM model. <br>
I think, the main reasons why it happens:<br> 
- short sentences in dataset (it also affects LSTM model, but not so destructive)<br>
- "bag of word" assumption. <br>
As I mentioned in "LSTM approach" part, there are other approaches for emotions detection in text.And it will be useful to study them.
