# 5559CHNMUS

## Overview
The topic of our project is Latent Space Visual Exploration for Traditional Chinese Music, where we extracted features from music using a Wavenet Autoencoder, visualized in our visualization system MusicLatentVIS, and conducted evaluation and analysis to the extracted latent vectors based on the visualization results.

I would like to introduce some backgrounds to our project for short. Feature extraction from music is an active research topic in the field of Music Information Retrieval (MIR), and the effectiveness of the extracted representation of music has got better thanks to the advance of deep learning. However, despite some existing efforts on visualizing music features[1], seldom do researchers conduct visual analysis to understand the latent vectors of music. Furthermore, most studies on MIR focus on western music, where traditional Chinese music, as a distinct music system played by a relatively smaller population, is often neglected. 

Therefore, we would like to extract features of traditional Chinese music, feed them into a visualization system, and see if we are able to draw insights from the visualization as music performers.

## Previous Work
Our project is a continuation of the work done by Shen et al. [2] In the previous work, the music is transformed into spectrograms, whose dimension is 501 by X. X is the number of time steps.  Each time step is 0.025-seconds. Shen et al. trained two autoencoders to extract latent vectors from spectrograms: a fully connected autoencoder for single time step feature extraction (presented as note latent vectors), and a LSTM autoencoder for a sequence of time steps (presented as segment latent vectors).

<figure>
    <img src='figure/FC-AE.png' />
    <figcaption>
	    <h6>Fig 1: Fully Connected Autoencoder for single column of spectrogram[2].</h6>
    </figcaption>
</figure>
---
![fig1](figure/FC-AE.png)

.center[Fig 1: Fully Connected Autoencoder for single column of spectrogram[2]]
---
<figure>
    <img src='figure/LSTM-AE.png' />
    <font size="2">
    <figcaption>
	    Fig 2: LSTM Autoencoder for sequence of columns of spectrogram. (A) is the architecture of the encoder. (B) is the architecture of decoder. (C) shows each layer in encoder takes a sequence of time steps and process one by one (the next depends on all previous) [2].
    </figcaption>
    </font>
</figure>

Shen et al. also designed a visualization system MusicLatentVIS with techniques such as t-SNE 2D projection, parallel coordinate, and heatmap. They applied t-SNE to both original data and note latent vectors  and compared them in 2D projection in order to explore whether note latent vectors with the same instrument/artist/pitch tend to form clusters as the note samples.
Fig 3: t-SNE 2D projection example. Left for original column vector. Right for the encoded version. Color encodes instruments, each individual dense cluster are the notes for a single pitch as labeled (e.g. Mi5, Fa4, etc.).

Parallel coordinate view is used to observe the value distribution in the latent representation. From the plot, the user is able to locate the dimension of the highest value for example, and the dimensions with no value.


## Project Goal
There are a couple of aspects that we hope to improve in the previous work. For example, in Fig. 5, each color represents a performer, and each point is a 10 seconds music segment. In spite of some obvious clusters, most points with different colors cluttered the middle part.

While troubled by this kind of observation, we hypothesized that it might be that the encoder is not strong enough to capture the intrinsic property of the music. 

Therefore we would like to improve the autoencoder model by building another one with Wavenet, a CNN architecture with shown superior performance dealing machine learning tasks involving audio [3]. In addition, we wanted to add more alternative 2D project views from different dimensionality reduction algorithms like PCA and UMAP, and finally, we would do an evaluation and analysis based on the visualization results from our Wavenet Autoencoder, and compare it to the previous work.

## Data Preprocessing
Our dataset contains 325 music files from 44 artists performing on 4 different instruments, which are Bamboo Flute (39 files), Erhu (70 files), Pipa (57 files), and Zheng (159 files). And the data is in the format of the spectrogram, where each column represents a time step of 0.025s and each row represents the energy on a particular frequency bin (there are 501 rows/bins in our spectrogram). An example spectrogram (not from our data) is shown below in Fig 6.

To boost the size of the training set and to accommodate our machine power for the training process, we did the following preprocessing steps:
* Randomly sample 100 starting points (a timestamp in the music) in each music file according to a uniform distribution. This results in 32500 data points in total.
* For each data point, from the starting point, take the following 29 time steps with an interval of 20 time steps to form a sequence. From our heuristic, an interval of 20 time steps (20 * 0.025s = 0.5s) will not break the naturality of the music. In this way, each data sample will be a sequence of 30 time steps representing a 14.5s (0.5s*(30-1)= 14.5s) slice from the original music.
* For the whole dataset, we did an amplitude-to-decibel operation because numbers in amplitude are usually small which are nor desired during the training.
* We further did an min-max normalization to map all decibel numbers into a range of [0, 1].
* To prevent the decoder from getting information from current time-step and any following time-step, we appended a dummy <start> point at the beginning of the sequence which will be the input to the decoder, and we appended a dummy <end> point at the end of the sequence which will be the output of the decoder. This is a normal practice in a sequence-to-sequence generative model because when the decoder tries to predict the value on the current time step, you do not want it to get any information from the current time step itself.
* Furthermore, to accommodate the implementation of convolutional layers in Keras, we also flipped the sequence (both input and output) in the decoder part. The input sequence to the encoder part remained unchanged. This is mainly because in Keras when the filter has an even filter size, in our case the filter size is always 2, the connection will be to the current and the next position in the input space, where in our case we want the connection to be to the current and the previous position. Thus, we flipped the sequence. And this will not conflict with any other idea of the original model design.

## Model
For a sequence-to-sequence autoencoder, one popular idea is the fully-dense neural network, where the encoder part has several dense layers and reduces the dimensionality of the sequence while the decoder part consists of several other dense layers and reconstructs the sequence. The dimension-reduced vector from the output of the encoder can be considered as an embedding of the whole sequence.

Another popular idea is the LSTM-based autoencoder, where the encoder has several LSTM layers and the hidden state of the last unit of the top LSTM layer is considered to be the ‘embedding’ and sent to the decoder. The decoder part is also some LSTM layers that take the ‘embedding’ as the input (or take both the ‘embedding’ and the shifted original sequence together as the input) and reconstruct the input sequence.

Both models have been explored by our teammate Rachel in her previous works. And in this work, we proposed to use a hybrid network as the autoencoder and tried to see if any improvement might be achieved.

The hybrid network still follows a standard encoder-decoder structure, where the encoder part consists of several LSTM layers (which is identical to the encoder in LSTM-based autoencoders) and the decoder part borrows ideas from the Wavenet (a network similar to PixelCNN that we have presented in the class). The hidden state (a vector) of the last LSTM unit in the top LSTM layer of the encoder can be considered as the “embedding” of the input music sequence. And it will be fed into the decoder as an additional input (in addition to the input sequence) that the decoder (Wavenet) can condition on. Assumably, the “embedding” should encode features of the music sequence.

An overview of Wavenet structure is shown in Fig. 7. In their original task, the authors used raw audio data where the input and the output will be a sequence of numbers (a 1-D vector where each position in the vector represents the information at one time step). Here, each dashed rectangle represents a layer in Wavenet. Within each layer, two filters (one normal filter and one gate filter) are formed to learn the feature of the previous layer’s output. The filters always have a size of 2 (in this 1-D case) but have a larger dilation rate (in the power of 2) as the layer goes deeper (as shown in Fig. 8). Also, the condition vector (the ‘embedding’ in our case) will be taken into consideration within the tanh and sigmoid unit. Residual connections and skip connections are also involved in order to allow a large number of layers to be trained properly. More details are referred to the original paper [3].

An overview of our autoencoder architecture is shown below in Fig 9. The encoder has 3 LSTM layers and the output of the last unit in the top layer will be considered as the ‘embedding’ and will be fed into the decoder. The input to the encoder will be the music sequences in spectrogram format which has a shape of (30, 501), and the output will be the embedding vector of shape (32). For the decoder, in general it follows the design of Wavenet. It takes the music sequence with the dummy <start> vector as the input and takes the music sequence with the dummy <end> vector as the output. Thus the input and the output will both have a shape of (31, 501). Also, the decoder takes the embedding vector as an additional input on which the decoding process will be conditioned. Inside each unit in the convolutional layers, the outputs of the previous layer (x) along with the embedding vector (h) will be computed according to the formula:

In our experiments, the weights for the embedding vector h will be shared within each time step (across all layers) but will remain unshared on different time steps. This may not be a good choice since different layers may learn different features from different levels. Sharing the weights across layers may hamper the learning process. But due to the time limit, we did not further experiment with this problem.

Furthermore, the main difference between our task and Wavenet’s original task is that our data is in the format of the spectrogram, meaning all the sequences will be in a format of an (n, m) matrix, where n is the length of the sequence (n_encoder = 30 and n_decoder = 31 in our case) and m is the number of frequency bins in the spectrogram (m = 501 in our case). Overall the architecture follows the Wavenet design but there are two noticeable choices that we can make in the experiment:
* **Extending Wavenet to multi-channel inputs**: to extend the wavenet to multi-channel (each time step is represented by a vector) data, we tried two ideas, namely Channel-dependent Filter Conv and Channel-integrated Filter Conv. For an input sequence of (31, 501), we first treated frequency bins as different channels and format the data to (31, 1, 501) to adapt the Keras implementation. For Channel-dependent Filter Conv, we learned one individual filter on each individual channel separately. In other words, each input to the conv layer will be separated to 501 * (31, 1, 1) slices and for each (31, 1, 1) slice, one filter (with filter size (2, 1) and an increasing dilation rate as the layer goes deeper) will be learned and still output a (31, 1, 1) tensor. All output slices will be concatenated together to still form a (31, 1, 501) tensor. This can be easily implemented by a function called DepthwiseConv2D in Keras. Another idea is that maybe each channel in a spectrogram should not be learned separately because each note in the music should only be represented by a combination of all frequency bins in the spectrogram. And this type of combination may have some intrinsic rules that should be looked at separately by the filter. For this reason, we tried Channel-integrated Filter Conv. The Channel-integrated Filter Conv layer takes in a (31, 1, 501) input and learns 501 filters where each filter looks at the input as a whole. And the output of the layer will still be (31, 1, 501). The filter size still remains as (2, 1) with an increasing dilation rate. This is a more common practice in convolutional neural networks and we actually found this produced more meaningful results.
* **Last output layer design**: for the last output layer, we tried two different settings, namely Last-conv and Last-sparse-dense. Last-conv follows the original Wavenet’s idea where the last output layer is a convolutional layer where the filter size is (1,1). Last-sparse-dense is that for each output unit, it connects to all previous time steps in a dense manner. In our implementation this will be connecting to all later units because of the sequence flipping. This type of sparse dense connection is shown in Fig 9 by the dash lines in the top layer.

We show the parallel coordinates and the t-SNE visualization of the results from 4 models here in Fig 10 - Fig 13. The four models are according to all possible combinations of the previous design choices. In Parallel coordinates, each embedding vector will be represented as a polyline from the left to the right. And the value on each position of the embedding vector is shown according to the y-axis. In t-SNE visualization, basically the embedding vectors are projected onto the 2d space and each embedding is shown as a point in the 2d space. As we can see from Fig 10-13, all embeddings in Channel-dependent Filter Conv + Last-conv only have one dimension that has value and the t-SNE visualization provides no use information. For Channel-dependent Filter Conv + Last-sparse-dense and Channel-integrated Filter Conv + Last-conv, more dimensions in the embedding vectors have actual values, while still a half of the dimensions still remain to be zero for all embeddings. Finally, for Channel-integrated Filter Conv + Last-sparse-dense, there are only four dimensions which are useless in the embeddings and the t-SNE visualization shows more meaningful results (some clear clusters). These visualizations simply show that Channel-integrated Filter Conv and Last-sparse-dense are better choices for our task.

## Links

Note Encoder: https://github.com/wangrqw/NoteEncoder.git

Data preprocessing: https://github.com/wangrqw/MUSDB.git

Visualization: https://github.com/wangrqw/VisBoard.git
