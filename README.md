# Abstract
Our task is to use a large amount of mp4 video files to create a model that can classify mp4 videos as either a deepfake or real video. This is a supervised classification problem. The dataset, [taken from this kaggle competition](https://www.kaggle.com/c/deepfake-detection-challenge), is 470gb of mp4 files (50 chunks of about 9gb each). Each mp4 file ranges from 4mb to 12mb. Each instance of a real video, is used to generate multiple deep fakes. The dataset is already divided into training data and test data. 

Our initial plan will be to classify each frame of a mp4 file as real or fake. Then taking the average of the k-frames and classifying whether it is real or fake. We will begin with processing only video frame data and determine if audio data improves the results of our model.  


## Example of the metadata.json provided for each chunk
```
{
  "vsghjdhss.mp4": {
    "label": "FAKE",
    "split": "train",
    "original": "id9edfjjs.mp4"
  },
  "isui3iwwu.mp4": {
    "label": "FAKE",
    "split": "train",
    "original": "fjhjhejkhj.mp4"
  },
  "vsghjdhss.mp4": {
    "label": "REAL",
    "split": "train",
  },
  ...
}
```
