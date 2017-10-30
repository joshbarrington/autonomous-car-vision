import cv2
import numpy
import mss


with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {'top': 40, 'left': 0, 'width': 800, 'height': 640}
    while 'Screen capturing':
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))

        # Display the picture
        cv2.imshow('OpenCV/Numpy normal', img)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
