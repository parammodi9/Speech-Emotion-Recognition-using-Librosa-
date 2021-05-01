# Speech Emotion Recognition
## Introduction
- This repository handles building and training Speech Emotion Recognition System.
- The basic idea behind this tool is to build and train/test a suited machine learning ( as well as deep learning ) algorithm that could recognize and detects human emotions from speech.
- This is useful for many industry fields such as making product recommendations, affective computing, etc.
## Requirements
- **Python 3.6+**
### Python Packages
- **librosa==0.6.3**
- **numpy**
- **pandas**
- **soundfile==0.9.0**
- **wave**
- **sklearn**
- **tqdm==4.28.1**
- **matplotlib==2.2.3**
- **pyaudio==0.2.11**
- **[ffmpeg](https://ffmpeg.org/) (optional)**: used if you want to add more sample audio by converting to 16000Hz sample rate and mono channel which is provided in ``convert_wavs.py``

Install these libraries by the following command:
```
pip3 install -r requirements.txt
```

### Dataset
You can download the dataset from the following link: https://drive.google.com/file/d/1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z7/view


### Emotions available
There are 8 emotions available: 
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
## Feature Extraction
Feature extraction is the main part of the speech emotion recognition system. It is basically accomplished by changing the speech waveform to a form of parametric representation at a relatively lesser data rate.

In this repository, we have used the most used features that are available in [librosa](https://github.com/librosa/librosa) library including:
- [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- Chromagram 
- MEL Spectrogram Frequency (mel)
- Contrast
- Tonnetz (tonal centroid features)

## Grid Search
Grid search results are already provided in `grid` folder, but if you want to tune various grid search parameters in `parameters.py`, you can run the script `grid_search.py` by:
```
python grid_search.py
```
This may take several minutes to complete execution, once it is finished, best estimators are stored and pickled in `grid` folder.

## Example 1: Using 3 Emotions
The way to build and train a model for classifying 3 emotions is as shown below:
```python
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob(r"speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        # print(file)
        file_name=os.path.basename(file)
        # print(file_name)
        emotion=emotions[file_name.split("-")[2]]
        # print(emotion)
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)        
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

```

```
![Screenshot (107)](https://user-images.githubusercontent.com/80505883/116776020-18002600-aa84-11eb-9188-2d1fc830d19f.png)
![Screenshot (110)](https://user-images.githubusercontent.com/80505883/116776021-19c9e980-aa84-11eb-890f-a4925bda4899.png)
![Screenshot (108)](https://user-images.githubusercontent.com/80505883/116776022-1a628000-aa84-11eb-99bb-b4222b4b4ccd.png)
![Screenshot (109)](https://user-images.githubusercontent.com/80505883/116776023-1afb1680-aa84-11eb-9b96-f70caaaaa7a4.png)
