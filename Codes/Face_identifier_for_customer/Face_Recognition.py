#Main
import cv2, sys, numpy, os
import time

def create_data():
    print('Enter Your Name')
    w = str(raw_input())
    haar_file = 'C:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml'
    datasets = 'C:\Real-time-face-recognition-in-python-using-opencv--master\datasets'  # All the faces data will be present this folder
    sub_data = 'C:\Real-time-face-recognition-in-python-using-opencv--master\datasets\\' + w  # These are sub data sets of folder, for my faces I've used my name

    path = os.path.join(datasets, sub_data)
    if not os.path.isdir(path):
        os.mkdir(path)
    (width, height) = (130, 100)  # defining the size of images

    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0)  # '0' is use for my webcam, if you've any other camera attached use '1' like this

    # The program loops until it has 60 images of the face.
    count = 1
    while count < 60:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            cv2.imwrite('%s/%s.png' % (path, count), face_resize)
        count += 1

        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break

def face_recognize():
    timeout = time.time() + 5
    size = 4
    haar_file = 'C:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml'
    datasets = 'C:\Real-time-face-recognition-in-python-using-opencv--master\datasets'
    # Part 1: Create fisherRecognizer
    print('Checking...')
    # Create a list of images and a list of corresponding names
    (images, lables, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(datasets, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                lable = id
                images.append(cv2.imread(path, 0))
                lables.append(int(lable))
            id += 1
    (width, height) = (130, 100)

    # Create a Numpy array from the two lists above
    (images, lables) = [numpy.array(lis) for lis in [images, lables]]

    # OpenCV trains a model from the images
    # NOTE FOR OpenCV2: remove '.face'
    model = cv2.face.createFisherFaceRecognizer()
    model.train(images, lables)

    # Part 2: Use fisherRecognizer on camera stream
    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0)
    while True:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            # Try to recognize the face
            prediction = model.predict(face_resize)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

            if prediction[1] < 500:
                cv2.putText(im, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                print('person verified : %s - %.0f') % (names[prediction[0]], prediction[1])
                exit()
            else :
                cv2.putText(im, 'not recognized', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if time.time() > timeout:
                print('Person not recognized')
                exit()
        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break

print('1) New User 2) Existing User')
choice = int(raw_input())
if (choice==1):
    create_data()
elif (choice==2) :
    face_recognize()
else :
    exit()
