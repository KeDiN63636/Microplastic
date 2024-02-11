import cv2
import numpy as np
from mb_main import data_create


def get_circle_mask(img, center=None, radius=None):
    height, width, depth = img.shape
    if center is None:
        center = (width // 2, height // 2)
    if radius is None:
        radius = min(height, width) // 4
    circle_mask = np.zeros((height, width), np.uint8)
    return cv2.circle(circle_mask, center, radius, 1, thickness=-1)


def mask_grayscale(img, mask, alpha=1.0, beta=100.0):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    inv_mask = (mask == 0).astype("int8")
    overlay = cv2.bitwise_and(img, img, mask=mask)
    background = cv2.bitwise_and(img_gray, img_gray, mask=inv_mask)
    out = background.copy()
    return cv2.addWeighted(overlay, alpha, out, beta, 0, out)


def markup(img):
    im = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 90, 120)
    gray2 = gray.copy()

    contours, hier = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if 1 < cv2.contourArea(cnt) < 5000:
            (x, y, w, h) = cv2.boundingRect(cnt)
            im = cv2.drawContours(im, [cnt], -1, (0, 255, 0), 1)
            # cv2.rectangle(gray2, (x, y), (x + w, y + h), 0, -1)
    return im


if __name__ == "__main__":
    deltaX = 80
    deltaY = 40
    img = cv2.imread("Datablet/train/1 (9).jpg")
    height, width, depth = img.shape
    mask = get_circle_mask(img, center=(width // 2 - deltaX, height // 2 + deltaY), radius=160)
    out = mask_grayscale(img, mask)
    out2 = markup(out)
    # plt.show(img)
    cv2.imshow("result", out)
    cv2.imshow("res", out2)
    data_create(out2, img)
    cv2.waitKey(0)
