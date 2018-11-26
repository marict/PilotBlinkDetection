# PilotBlinkDetection
Blink Detection of airplane pilots during takeoff and landing in flight simulator  
  
Based off of this blog post by Adrian Roseblock:  
https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/  

Installation Instructions (Windows 10):
1. Download Anaconda
    ..*Config that worked for me: 
    ..*Python 3.7 version
    ..*Install for all users
    ..*Choose installation directory as {ANYTHING}/Anaconda3
    ..*Update path manually
2. Connect to conda-forge channel
    ..*conda config --add channels conda-forge 
3. Create new environment (default has too many conflicts) and activate it
    ..*conda create --name myenv
    ..*activate myenv
4. conda install dlib opencv scipy imutils scikit-learn pandas
5. Pip install imutils
