# Kor-Sentence-Classification

## Sentence Classification for Korean

### Models
- Character Level CNN
    - [Character-level Convolutional Networks for Text Classification, X. Zhang, J. Zhao, Y. LeCun](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
    - Little Modification
    - Named 'Char_CNN'
- Charcater Level CNN 2
    - [Convolutional Neural Networks for Sentence Classification, Y. Kim](http://www.aclweb.org/anthology/D14-1181)
    - Little Modificataion
    - Named 'Wide' 
- VDCNN
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition, K. Simonyan, A. Zisserman](https://arxiv.org/abs/1409.1556)
    - Modified for Sentence Classification
    - Named 'VDCNN'
- LSTM
    - Basic LSTM Model
    - Predict using Fully connected layers 
- Multilayer LSTM
    - Multilayer LSTM

### Details
- Preprocessing
    * Character Level (음소 or 음절)
    * Digits and Specials
    * For eumjeol(Syllable), use frequent 2350
- Training
    * Half learning rate every 3 epochs
    * 

- Configuration
    * 'main.py'     : main run file
    * '--output'    : # of Output
    * '--epochs'    : # of training epochs
    * '--batch'     : Batch Size
    * '--lr'        : Learning rate
    * '--strmaxlen' : Maximum Limit of String Length
    * '--charSize'  : Vocab Size
    * '--rnn_hidden': Hidden Dimension for LSTMs
    * 'filter_num'  : # of Filter of one CNN Filter
    * '--emb'       : Embedding Dimension
    * '--eumjeol'   : Use Eumjeol(Syllable-level) if specified 
    * '--bi'        : Use Bi-directional if specified
    * '--model'     : Model Selection (CHAR, WIDE, VDCNN, LSTM, MULTI_LSTM)

### To Run
- Set FC,layer and RNN layers in 'main.py'
- run 'main.py' with arguments as you wish 