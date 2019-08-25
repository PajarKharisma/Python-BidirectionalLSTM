<h2 align="center">Long sort term memory</h2>
This program implements lstm methode for classfication Sentiment-Analysis about hate speech and non hate speech in twitter. Dataset uses csv format file that contains collection of tweets about politics.  
Before trained, dataset will pass preprocess. Preprocess will remove unnecessary word, separate label and data, and change dataset from string to numeric because training process only accept numeric format.  

### Requirement :
- Python 3
- Numpy
- Matplotlib
- Scikit-learn
- Tensorflow
- Keras
<br><br>

### Screenshot :
<img src="img/Figure_1.png">
<br><br>

### How to install?
1. Clone this project `git clone https://github.com/PajarKharisma/Python-BidirectionalLSTM.git`

2. Install all requirements. You can type `pip install -r requirements.txt` to install it.

3. copy `file/indonesian-sentimen` to `C:\\Users\\LiSA\\AppData\\Roaming\\nltk_data\\corpora\\stopwords\\`

4. For running, type `python main.py` inside src/main folder
<br><br>

### References
- https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/
- http://www.insightsbot.com/blog/1wAqZg/keras-lstm-example-sequence-binary-classification
- https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
- https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/