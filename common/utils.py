import cv2
import matplotlib.pyplot as plt


def show_image(image: cv2.Mat, title: str = "No title"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_full_image_plot(image: cv2.Mat, code: int = cv2.COLOR_BGR2RGB, dpi: int = 80):
    height, width, _ = image.shape
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax1 = fig.add_axes((0, 0, 1, 1))
    ax1.imshow(cv2.cvtColor(image, code))
    plt.show()


def show_image_plot(image: any):
    fig = plt.figure()
    ax1 = fig.add_axes((0, 0, 1, 1))
    ax1.imshow(image)
    plt.show()
