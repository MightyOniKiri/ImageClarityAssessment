import cv2


def laplacian_score(image):
    """输入图片，返回拉普拉斯方差分数。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(score)


def tenengrad_score(image):
    """输入图片，返回 Tenengrad 分数。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    score = (gx ** 2 + gy ** 2).mean()
    return float(score)