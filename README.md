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
<h6>Fig 4: Segment Latent Vector Visualization.</h6>
</figcaption>

While troubled by this kind of observation, we hypothesized that it might be that the encoder is not strong enough to capture the intrinsic property of the music. 

Therefore we would like to improve the autoencoder model by building another one with Wavenet, a CNN architecture with shown superior performance dealing machine learning tasks involving audio [3]. In addition, we wanted to add more alternative 2D project views from different dimensionality reduction algorithms like PCA and UMAP, and finally, we would do an evaluation and analysis based on the visualization results from our Wavenet Autoencoder, and compare it to the previous work.


## Data Preprocessing
Our dataset contains 325 music files from 44 artists performing on 4 different instruments, which are Bamboo Flute (39 files), Erhu (70 files), Pipa (57 files), and Zheng (159 files). And the data is in the format of the spectrogram, where each column represents a time step of 0.025s and each row represents the energy on a particular frequency bin (there are 501 rows/bins in our spectrogram). An example spectrogram (not from our data) is shown below in Fig 6.

<img src='figure/spectrogram.png'>
<figcaption>
<h6>Fig 5: An example of a spectrogram</h6>
</figcaption>

To boost the size of the training set and to accommodate our machine power for the training process, we did the following preprocessing steps:
* Randomly sample 100 starting points (timestamps) in each spectrogram according to a uniform distribution. This sampling operation gives us 32500 data points in total.
* From each starting point, take the following 29 time steps with an interval of 20 time steps to form a sequence. From our heuristics, an interval of 20 time steps (20 * 0.025s = 0.5s) will not break the naturality of the music. In this way, each data sample will be a sequence of 30 time steps representing a 14.5s (0.5s*(30-1)= 14.5s) slice from the original music.
```python
def load_data(num_samples=100, num_timesteps=30, timestep_length=20):
    sample_list = []
 
    for instrument in os.listdir(data_path):
        instrument_path = os.path.join(data_path, instrument)
        for artist in os.listdir(instrument_path):
            artist_path = os.path.join(instrument_path, artist)
            for music_file in os.listdir(artist_path):
                this_music_ndarray = np.load(os.path.join(artist_path, music_file))
                this_music_ndarray = np.moveaxis(this_music_ndarray, 0, 1)
                music_length = this_music_ndarray.shape[0]
                # randomly take num_samples starting points
                starting_points = np.random.uniform(0, music_length - timestep_length * num_timesteps, num_samples).astype('int')
                for start_point in starting_points:
                    sample_list.append({'Instrument': instrument,
                                        'Artist': artist,
                                        'Music File Name': music_file,
                                        'Starting Index:': start_point.item(),
                                        'Ending Index': (start_point + timestep_length * (num_timesteps - 1)).item(),
                                        'Interval': timestep_length,
                                        # take sequence with intervals
                                        'Spec Array': this_music_ndarray[start_point: start_point + timestep_length *num_timesteps: timestep_length, :]})
    return sample_list
```
* For the whole dataset, we converted the amplitude values to decibel because numbers in amplitude are usually small that are not  desired during the training.
* We further did an min-max normalization to map all decibel numbers in a range of [0, 1].
* To prevent the decoder from getting information from current time-step and any following time-step, we appended a dummy <start> point at the beginning of the sequence which will be the input to the decoder, and we appended a dummy <end> point at the end of the sequence which will be the output of the decoder. This is a normal practice in a sequence-to-sequence generative model because when the decoder tries to predict the value on the current time step, people do not expect it to get any information from the current time step itself.
*Furthermore, we also flipped the sequence (both input and output) in the decoder part to accommodate the implementation of convolutional layers in Keras. The input sequence to the encoder part remained unchanged. This is mainly because in Keras when the filter has an even filter size, in our case the filter size is always 2, the connection will be to the current and the next position in the input space, where in our case we want the connection to be to the current and the previous position. Thus, we flipped the sequence. And this will not conflict with any other idea of the original model design.
```python
def get_raw_sequence(json_list):
    input_en = []
    input_de = []
    output_de = []
 
    # define start/end dummy vectors
    dummy_start_vec = np.zeros((1, feature_dim))
    dummy_end_vec = np.zeros((1, feature_dim))
    dummy_start_vec[0, 0] = 1.0
    dummy_end_vec[0, -1] = 1.0
 
    # get min/max value from all decibel numbers
    min_db = np.inf
    max_db = - np.inf
    for item in json_list:
        this_seq = item['Spec Array']
        db_seq = amplitude_to_db(this_seq, ref=1.0)
 
        input_en.append(db_seq)
        min_db = min(min_db, db_seq.min())
        max_db = max(max_db, db_seq.max())
 
    for index, item in enumerate(input_en):
        # min-max normalization
        input_en[index] = (item - min_db) / (max_db - min_db)
        # flip the sequence for decoder
        this_input_de = np.flip(input_en[index], 0)
        # append dummy point
        input_de.append(np.append(this_input_de, dummy_start_vec, axis=0))
        output_de.append(np.append(dummy_end_vec, this_input_de, axis=0))
 
    # input of the encoder
    input_en = np.array(input_en)
    # input of the decoder
    input_de = np.array(input_de)
    # output of the decoder
    output_de = np.array(output_de)
 
    return input_en, input_de, output_de
```
## Model


## Visualization System
Visualization is helpful for both understanding data and debugging models. Our visualization panel has three basic parts, i.e. the overview panel (home page), the Clip Visualization Panel (ClipVis page), and the Note Visualization Panel (NoteVis page). By clicking the tab on the navigation bar, people can switch between different panels. The selected tab would be highlighted on the navigation bar. The embedding results of different models can be selected through the uploading box on the right corner. 

### Overview Panel
Overview panel is specifically for understanding the data. When data is imported into the visualization system, the overview panel would display a parallel plot to show every dimension of the embedding result. This view gives people a glance of the embedding. 

### Clip Visualization Panel
We adopt three methods to project our ‘embedding’ into 2D space, i.e. t-SNE, PCA and UMAP. Both t-SNE and UMAP model the similar objects as nearby points, and dissimilar ones as distant points; however, since the two methods work in different schemes, the clusters may look different when projecting to 2D space; for example, our result projected by UMAP usually looks more tight. Hence, making comparison of results of both t-SNE and UMAP is probably helpful to understand the data better. PCA reduces the dimension of ‘embedding’ by decomposing the covariance matrix of the data, so it preserves the distance in the original high dimensional space. 

The Clip Visualization Panel has three sub panels for displaying the three projections above. From the top to the bottom, they are t-SNE view, PCA view, and UMAP view. Each view has five sub displays. The major one is the overview display, which shows all the clips embedding from all four instruments and all the artists. Each of the four small panels stands for one instrument; for instance, the middle two are for BambooFlute and Erhu, and the right two are for Pipa and Zheng. 

The figure is a visualization of one of our embedding results. From the top to the bottom, the ClipVis 

### Note Visualization Panel
The Note Visualization Panel incorporated 2D projection from t-SNE and PCA as well as a parallel coordinate sub panel. The note latent vector for this panel comes from the fully connected autoencoder from Shen et al. We reworked the interface and developed a fisheye magnifier effect for the t-SNE projection.

Overall, the panel is formatted in the order of t-SNE projection, PCA projection, and then the parallel coordinates.

Fig X: Note Visualization Panel (2/3)

Fig. X: Note Visualization Panel (3/3)

In the t-SNE sub panel, before the magnifier is applied, it’s hard to check the distances between each note latent vectors in the same cluster of pitch and instrument (e.g. Mi4 in red), but with the magnifier we have a better view of the intra-cluster distances.

Fig. X: Fisheye magnifier effect for t-SNE projection. Left: original. Right: Magnified.


## Observation: Compare the LSTM Vis and WaveNet Vis
Left: the result of LSTM Autoencoder. Right: the result of WaveNet Autoencoder. 

From fig x above, we can clearly see that the LSTM Autoencoder has better clustering results on both t-SNE and UMAP view. One thing to be clarified here is why the result in the previous work does not look good. The reason is that in the previous work, the points are colored by the artists whose number is above 40, but 40 colors are distinctive in human eyes. This time, we upgraded the visual encoding: we marked music segments played by different instruments by unique symbols and then assigned colors to artists who play the same instruments. To be more clear, we also make four subplots for the four instruments. In this way, we can see the segment played by the same artist more easily. 


## Future Works
Several points are remained to be in the future work from this project:
Look into why some dimensions were not getting any information in each model during training.
In the wavenet decoder, try not to share the weights for the hidden vector across different layers.
Although we experimented with different model settings to find a better choice, the hyperparameters of the model remained unexplored in this project. An exploration into the hyperparameter space may lead to a better performance.


## Links

Note Encoder: https://github.com/wangrqw/NoteEncoder.git

Data preprocessing: https://github.com/wangrqw/MUSDB.git

Visualization: https://github.com/wangrqw/VisBoard.git
