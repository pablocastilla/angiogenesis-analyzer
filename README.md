# Angiogenesis analyzer
This project is done for the Biomedicine, Biotechnology and Public Healthcare of the University of Cádiz. They idea is to make easier to process the microscopic images of their experiments related to angiogenesis.

##Process
- Each tif has different frames in it and each frame is processed doing the following:
1. Extract the frame
2. Resize the frame
3. Contours are found using opencv
4. Image is skeletonized
5. Inner graph is found using the skeleton and the contours
6. Selected measures are calculated and returned (number of joins, number of meshes, total meshes area, number of segments, total segments length)

- A CSV is created and returned with the results of each frame

##Work with the code
###Create environment
1. Install python 3.7 amd64 adding it to the path
2. https://docs.python.org/3/library/venv.html
3. python -m venv ./venv
4. activate
5. pip install final-requirements.txt

### Create executable
From the terminal with the environment activated type
- pyinstaller --onefile process_tiff.py


