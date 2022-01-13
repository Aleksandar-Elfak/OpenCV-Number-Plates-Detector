import cv2
import numpy as np

height = 480
width = 854


def preprocessing(img):
    Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Blur = cv2.GaussianBlur(Gray, (5, 5), 1)
    Canny = cv2.Canny(Blur, 0, 20)
    kernel = np.ones((5, 5))
    Dial = cv2.dilate(Canny, kernel, iterations=2)
    Threshold = cv2.erode(Dial, kernel, iterations=1)
    # cv2.imshow("Plate1", Gray)
    # cv2.imshow("Plate2", Blur)
    # cv2.imshow("Plate3", Canny)
    # cv2.imshow("Plate4", Dial)
    # cv2.imshow("Plate5", Threshold)
    return Threshold


def getContours(img, coloredImg, multiple=False):
    main = np.array([])
    maxArea = 0
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours[0]:
        area = cv2.contourArea(contour)
        if area > 50000:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if area > maxArea and len(approx) == 4 if multiple is False else True:
                main = approx
                maxArea = area
    # cv2.drawContours(coloredImg, main, -1, (255, 0, 0), 20)
    # cv2.imshow("Test", coloredImg)
    return main


def reorderPoints(points):
    reordered = list([None, None, None, None])

    reordered[0] = min(points, key=lambda x: x[0][0] + x[0][1])

    reordered[1] = max(points, key=lambda x: x[0][0] - x[0][1])

    reordered[2] = min(points, key=lambda x: x[0][0] - x[0][1])

    reordered[3] = max(points, key=lambda x: x[0][0] + x[0][1])

    return reordered


def zoomIn(img, points):
    perspective = cv2.getPerspectiveTransform(
        np.float32(points),
        np.float32([[0, 0], [width, 0], [0, height], [width, height]]),
    )
    warped = cv2.warpPerspective(img, perspective, (width, height))

    warped = cv2.resize(warped, (width, height))

    return warped


def showPlate(imgOriginal, size, multiple=False):
    imgCropped = imgOriginal[size[1] : size[1] + size[3], size[0] : size[0] + size[2]]
    imgCropped = cv2.resize(imgCropped, (width, height))
    processedImg = preprocessing(imgCropped)
    cont = getContours(processedImg, imgCropped, multiple)
    if len(cont) > 0:
        newPoints = reorderPoints(cont)
        zoomed = zoomIn(imgCropped, newPoints)
        # cv2.imshow("Plate", zoomed)
        return zoomed
    return None


def Image():
    carPlateCascade = cv2.CascadeClassifier(
        "venv\\Resources\\haarcascade_russian_plate_number.xml"
    )

    images = [
        cv2.imread("venv\\Resources\\car1.png"),
        cv2.imread("venv\\Resources\\car2.png"),
        cv2.imread("venv\\Resources\\car3.jpg"),
        cv2.imread("venv\\Resources\\car4.jpg"),
    ]

    print("Press any key for next Image.")

    for img in images:
        img = cv2.resize(img, (width, height))
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        plates = carPlateCascade.detectMultiScale(imgGray, 1.1, 4)

        size = None

        imgOriginal = img.copy()

        for (x, y, w, h) in plates:
            size = (x, y, w, h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            break

        plate = None

        if size is not None:
            plate = showPlate(imgOriginal, size, True)

        imgJoined = np.hstack((plate, img))
        cv2.imshow("Car", imgJoined)
        cv2.waitKey(0)


def Video():
    carPlateCascade = cv2.CascadeClassifier(
        "venv\\Resources\\haarcascade_russian_plate_number.xml"
    )

    print(carPlateCascade)

    print("Press q to stop the video.")

    video = cv2.VideoCapture("venv\\Resources\\CarVideo1_720.mp4")

    plate = np.zeros((height, width, 3), np.uint8)

    while True:
        success, img = video.read()
        img = cv2.resize(img, (width, height))
        plates = carPlateCascade.detectMultiScale(img, 1.1, 4, maxSize=(200, 200))
        size = None
        imgOriginal = img.copy()

        for (x, y, w, h) in plates:
            size = (x, y, w, h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if size is not None:
            tmp = showPlate(imgOriginal, size)
            if tmp is not None:
                plate = tmp

        imgJoined = np.hstack((plate, img))
        cv2.imshow("Car", imgJoined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def start():
    mode = input("For video enter V, for Images enter I: ")
    if mode == "V":
        Video()
    elif mode == "I":
        Image()


start()
