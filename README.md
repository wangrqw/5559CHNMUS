# 5559CHNMUS

## Overview
The topic of our project is Latent Space Visual Exploration for Traditional Chinese Music, where we extracted features from music using a Wavenet Autoencoder, visualized in our visualization system MusicLatentVIS, and conducted evaluation and analysis to the extracted latent vectors based on the visualization results.

I would like to introduce some backgrounds to our project for short. Feature extraction from music is an active research topic in the field of Music Information Retrieval (MIR), and the effectiveness of the extracted representation of music has got better thanks to the advance of deep learning. However, despite some existing efforts on visualizing music features[1], seldom do researchers conduct visual analysis to understand the latent vectors of music. Furthermore, most studies on MIR focus on western music, where traditional Chinese music, as a distinct music system played by a relatively smaller population, is often neglected. 

Therefore, we would like to extract features of traditional Chinese music, feed them into a visualization system, and see if we are able to draw insights from the visualization as music performers.

## Previous Work
Our project is a continuation of the work done by Shen et al. [2] In the previous work, the music is transformed into spectrograms, whose dimension is 501 by X. X is the number of time steps.  Each time step is 0.025-seconds. Shen et al. trained two autoencoders to extract latent vectors from spectrograms: a fully connected autoencoder for single time step feature extraction (presented as note latent vectors) shown in Fig 1, and a LSTM autoencoder for a sequence of time steps (presented as segment latent vectors) shown in Fig 2.

<p>
<img src='figure/FC-AE.png' width=500>
<figcaption>
  <h6>Fig 1: Fully Connected Autoencoder for single column of spectrogram[2].</h6>
</figcaption>
<img src='figure/LSTM-AE.png' width=500>
<figcaption>
  <h6>Fig 2: LSTM Autoencoder for sequence of columns of spectrogram. (A) is the architecture of the encoder. (B) is the architecture of decoder. (C) shows each layer in encoder takes a sequence of time steps and process one by one (the next depends on all previous) [2].</h6>
</figcaption>
</p>

Shen et al. also designed a visualization system MusicLatentVIS with techniques such as t-SNE 2D projection, parallel coordinate, and heatmap. They applied t-SNE to both original data and note latent vectors  and compared them in 2D projection in order to explore whether note latent vectors with the same instrument/artist/pitch tend to form clusters as the note samples.
<p>
<img src='figure/t-SNE_example.png'>
<figcaption>
<h6>Fig 3: t-SNE 2D projection example. Left for original column vector. Right for the encoded version. Color encodes instruments, each individual dense cluster are the notes for a single pitch as labeled (e.g. Mi5, Fa4, etc.).</h6>
</figcaption>
</p>
Parallel coordinate view is used to observe the value distribution in the latent representation. As shown in Fig 4, the user is able to locate the dimension of the highest value for example, and the dimensions with no value.


## Project Goal
There are a couple of aspects that we hope to improve in the previous work. For example, in Fig. 5, each color represents a performer, and each point is a 10 seconds music segment. In spite of some obvious clusters, most points with different colors cluttered the middle part.

<img src='figure/oldLSTMvis.png'>
<figcaption>
<h6>Fig 5: Segment Latent Vector Visualization.</h6>
</figcaption>

While troubled by this kind of observation, we hypothesized that it might be that the encoder is not strong enough to capture the intrinsic property of the music. 

Therefore we would like to improve the autoencoder model by building another one with Wavenet, a CNN architecture with shown superior performance dealing machine learning tasks involving audio [3]. In addition, we wanted to add more alternative 2D project views from different dimensionality reduction algorithms like PCA and UMAP, and finally, we would do an evaluation and analysis based on the visualization results from our Wavenet Autoencoder, and compare it to the previous work.


## Data Preprocessing


## Model


## Links

Note Encoder: https://github.com/wangrqw/NoteEncoder.git

Data preprocessing: https://github.com/wangrqw/MUSDB.git

Visualization: https://github.com/wangrqw/VisBoard.git
