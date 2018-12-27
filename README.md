# PilotBlinkDetection
Blink Detection of airplane pilots during takeoff and landing in flight simulator  
  
Based off of this blog post by Adrian Roseblock:  
https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/  

Installation Instructions (Windows 10):
1. Download Anaconda
    <br />&nbsp;Config that worked for me: 
    <br />&nbsp;Python 3.7 version
    <br />&nbsp;Install for all users
    <br />&nbsp;Choose installation directory as {ANYTHING}/Anaconda3
    <br />&nbsp;Update path manually
2. Connect to conda-forge channel
    <br />&nbsp;conda config --add channels conda-forge 
3. Create new environment (default has too many conflicts) and activate it
    <br />&nbsp;conda create --name myenv
    <br />&nbsp;activate myenv
4. conda install dlib keras opencv scipy scikit-learn pandas
5. Pip install imutils
6. You should be good to go! 
  <br />&nbsp;I like to run my python through Notepad++, to do this create run macro with 
  <br />&nbsp;activate dlib_env && python -i "$(FULL_CURRENT_PATH)"
