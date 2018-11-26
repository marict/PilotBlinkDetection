# PilotBlinkDetection
Blink Detection of airplane pilots during takeoff and landing in flight simulator  
  
Based off of this blog post by Adrian Roseblock:  
https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/  

Installation Instructions (Windows 10):
1. Download Anaconda
    \n\tConfig that worked for me: 
    \n\tPython 3.7 version
    \n\tInstall for all users
    \n\tChoose installation directory as */Anaconda3
    \n\tUpdate path manually
2. Connect to conda-forge channel
    \n\tconda config --add channels conda-forge 
3. Create new environment (default has too many conflicts) and activate it
    \n\tconda create --name myenv
    \n\tactivate myenv
4. conda install dlib opencv scipy imutils scikit-learn pandas
5. Pip install imutils
