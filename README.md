# PilotBlinkDetection
Blink Detection of airplane pilots during takeoff and landing in flight simulator  
  
Based off of this blog post by Adrian Roseblock:  
https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/  

Installation Instructions (Windows 10):
1. Download Anaconda
    &nbsp;Config that worked for me: 
    &nbsp;Python 3.7 version
    &nbsp;Install for all users
    &nbsp;Choose installation directory as {ANYTHING}/Anaconda3
    &nbsp;Update path manually
2. Connect to conda-forge channel
    &nbsp;conda config --add channels conda-forge 
3. Create new environment (default has too many conflicts) and activate it
    &nbsp;conda create --name myenv
    &nbsp;activate myenv
4. conda install dlib opencv scipy imutils scikit-learn pandas
5. Pip install imutils
