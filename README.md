# 5559CHNMUS

## Overview
Our project topic is Latent Space Visual Exploration for Traditional Chinese Music, where we extracted features from music using a Wavenet Autoencoder, visualized in our visualization system MusicLatentVIS, and conducted evaluation and analysis to the extracted latent vectors based on the visualization results.

I would like to introduce some backgrounds to our project here. Feature extraction from music is an active research topic in the field of Music Information Retrieval (MIR), and the effectiveness of the extracted representation of music has got better thanks to the advance of deep learning. However, despite some existing efforts on visualizing music features[1], seldom do researchers conduct visual analysis to understand the latent vectors of music. Furthermore, most studies on MIR focus on western music, where traditional Chinese music, as a distinct music system played by a relatively smaller population, is often neglected. 

Therefore, we would like to do our own feature extraction on traditional Chinese music, feed them into a visualization system, and see if we, as music practitioners, are able to draw insights from the visualization.

## Previous Work
Our project is a continuation of the work done by Shen et al. [2] In their work, the music is transformed into spectrograms, where the dimension is 501 by X, X is the number of time steps with 0.025 seconds each. They trained two autoencoders to extract latent vectors from spectrograms: a fully connected autoencoder for single time step feature extraction, and a LSTM autoencoder for a sequence of time steps.




## Links

Naive Previous Work: https://github.com/wangrqw/NoteEncoder.git

Data preprocessing: https://github.com/wangrqw/MUSDB.git

Visualization: https://github.com/wangrqw/VisBoard.git
