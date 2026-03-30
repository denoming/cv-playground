import cv2 as cv
import matplotlib.pyplot as plt


def show_image(image: cv.Mat, title: str = "No title"):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def show_full_image_plot(image: cv.Mat, title: str = "No title", code: int = cv.COLOR_BGR2RGB, dpi: int = 80):
    height, width, = image.shape
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax1 = fig.add_axes((0, 0, 1, 1))
    ax1.set_title(title)
    ax1.imshow(cv.cvtColor(image, code))
    plt.show()