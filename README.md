# ACCirO

We adapted color based and geometry data extraction of Pie Charts, Scatter plot and it's varients(Simple, Dot, Bubble) in this system.  

## Text Detection and Recognition Module

This module performs text detection and recognition on chart Image. We use a deep-learning-based OCR, namely Character Region Awareness for Text Detection, CRAFT | [Paper](https://arxiv.org/abs/1904.01941) | [Code](https://github.com/clovaai/CRAFT-pytorch) | succeeded by a scene text recognition framework, STR | [Paper](https://arxiv.org/abs/1904.01906) | [Code](https://github.com/clovaai/deep-text-recognition-benchmark) |
 
### To run the code

Things to be taken care before runing the code:
1. Download the pretrained model [craft_mlt_25k.pth](https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view), and place model at the following path ```ChartDecode/CRAFT_TextDetector/craft_mlt_25k.pth```
2. Download the pretrained model [TPS-ResNet-BiLSTM-Attn.pth](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW), tand place model at the following path ```ChartDecode/Deep_TextRecognition/TPS-ResNet-BiLSTM-Attn.pth```
3. The code is developed and tested on Python 3.6 you can also find attached requirements.txt to avoid errors due to compatibility issues
4. Finally you can run the  ```main.py``` file and provide the path of your chart image file.
    It generates the following files as output:
    1. data_```filename```.csv: contains extracted data values along with additional semantic attributes like chart_type, title, x-title, and y-title that helps in chart reconstruction and summarization
    2. Reconstructed_```filename```.png: The reconstructed image from extracted  data_```filename```.csv file.
    3. summ_```filename```.txt: The chart text summary generated using templated-NLG approach based on our user-study observations
5. Also find the synthetically genrerated test data set for this system with it's results at ```ChartDecode/SYNTHETIC_DATA```.


