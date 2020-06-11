import cv2
import numpy as np
import operator
import keras
import solve_sudoku
import pytesseract


def preprocess_img(image, dilate_single_digit):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur
    blur_image = cv2.GaussianBlur(gray, (3, 3), 0)
    show_image(blur_image, "blur")

    # Threshold, adaptive to be light independant
    thresh = cv2.adaptiveThreshold(
        blur_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    show_image(thresh, "thresh")

    # Bitwise_not
    thresh_inv = cv2.bitwise_not(thresh)
    show_image(thresh_inv, "bitwise_not")
    if dilate_single_digit:
        kernel = np.ones((1, 1))
        dilated = cv2.dilate(thresh_inv, kernel)
        show_image(dilated, "dilate")
        return thresh_inv
    # Dilate to fill in "holes"
    kernel = np.ones((2, 2))
    dilated = cv2.dilate(thresh_inv, kernel)
    show_image(dilated, "dilate")

    return dilated


def find_corners(image):
    """
    Find the corners of the sudoko but finding the largest contour
    """
    contours, h = cv2.findContours(
        image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    contours = sorted(contours, key=cv2.contourArea,
                      reverse=True)  # Sort by area, descending
    polygon = contours[0]  # Largest polygon
    bottom_right, _ = max(
        enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1]
                                 for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(
        enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1]
                                  for pt in polygon]), key=operator.itemgetter(1))

    # Return an array of all 4 points using the indices
    # Each point is in its own array of one coordinate
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def display_points(in_img, points, radius=5, colour=(0, 0, 255)):
    """Draws circular points on an image."""
    img = in_img.copy()

    # Dynamically change to a colour image if necessary
    if len(colour) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for point in points:
        img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
    # show_image(img, "corners")
    return img


def show_image(img, title):
    """Shows an image until any key is pressed"""
    cv2.imshow(title, img)  # Display the image
    # Wait for any key to be pressed (with the image window active)


def distance_between(p1, p2):
    """Returns the scalar distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp(img, crop_rect):
    """Crops and warps a rectangular section from an image into a square of similar size."""

    # A rectangle described by top left, top right, bottom right and bottom left points
    top_left, top_right, bottom_right, bottom_left = crop_rect[
        0], crop_rect[1], crop_rect[2], crop_rect[3]

    rect = np.array([top_left, top_right, bottom_right,
                     bottom_left], dtype='float32')

    # Find the longest side in the rectangle
    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])

    # Create a square with side of the calculated length, this is the new perspective we want to warp to
    square = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1],
                       [0, side - 1]], dtype='float32')

    # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    m = cv2.getPerspectiveTransform(rect, square)

    # Performs the transformation on the original image and returns
    return cv2.warpPerspective(img, m, (int(side), int(side)))


def image_size(img):
    """
    Gets the size of an image eg. 350x350
    """
    return tuple(img.shape[1:: -1])


def infer_grid(img):
    """Infers 81 cell grid from a square image."""
    squares = []
    side = img.shape[: 1]
    side = side[0] / 9
    for i in range(9):
        for j in range(9):
            p1 = (i * side, j * side)  # Top left corner of a bounding box
            # Bottom right corner of bounding box
            p2 = ((i + 1) * side, (j + 1) * side)
            squares.append((p1, p2))
    for square in squares:
        img = cv2.rectangle(img, tuple(int(x) for x in square[0]), tuple(
            int(x) for x in square[1]), (0, 255, 0))
    show_image(img, "grid")
    return squares


def cut_from_rect(img, rect):
    """Cuts a rectangle from an image using the top left and bottom right points."""
    return img[int(rect[0][1]): int(rect[1][1]), int(rect[0][0]): int(rect[1][0])]


def scale_and_centre(img, size, margin=0, background=0):
    """Rescales and centre an image onto a new background square."""
    h, w = img.shape[: 2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad,
                             r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))


def find_largest_feature(img, scan_tl=None, scan_br=None):
    """
    Uses `floodFill` function to find a bounding box of the area it filled to find the biggest
    connected pixel structure in the image. 
    """
    img = img.copy()
    height, width = img.shape[:2]

    max_area = 0
    seed_point = (None, None)

    if scan_tl is None:
        scan_tl = [0, 0]

    if scan_br is None:
        scan_br = [width, height]

    # Loop through the image
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            # Only fill light or white squares
            if img.item(y, x) == 255 and x < width and y < height:
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                    max_area = area[0]
                    seed_point = (x, y)

    # Colour everything grey (compensates for features outside of our middle scanning range
    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)

    # A Mask that is 2 pixels bigger than the image for padding
    mask = np.zeros((height + 2, width + 2), np.uint8)

    # fill the main feature
    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 64:  # Fill  with black anything that isn't the main feature
                cv2.floodFill(img, mask, (x, y), 0)

            # Find the bounding parameters
            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    return img, np.array(bbox, dtype='float32'), seed_point


def extract_digit(img, rect, size):
    """Extracts a digit (if one exists) from a Sudoku square."""

    digit = cut_from_rect(img, rect)  # Get the digit box from the whole square

    height, width = digit.shape[:2]
    margin = int(np.mean([height, width]) / 2.5)
    _, bbox, seed = find_largest_feature(
        digit, [margin, margin], [width - margin, height - margin])
    digit = cut_from_rect(digit, bbox)

    # Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
    width = bbox[1][0] - bbox[0][0]
    height = bbox[1][1] - bbox[0][1]

    # Ignore any small bounding boxes
    if width > 0 and height > 0 and (width * height) > 100 and len(digit) > 0:
        return scale_and_centre(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)


def get_digits(img, squares, size):
    """Extracts digits from their cells and builds an array"""
    digits = []
    img = preprocess_img(img.copy(), dilate_single_digit=True)
    show_image(img, "test")
    for square in squares:
        digits.append(extract_digit(img, square, size))
    return digits


def find_sudoku(img):
    """
    Entrypoint to find the digits of sudoku for the model to interpret
    """
    process_img = preprocess_img(img, dilate_single_digit=False)
    corners = find_corners(process_img)
    display_points(sudoku, corners)
    cropped_warped = crop_and_warp(sudoku, corners)
    # show_image(cropped_warped, "warp")
    digit_squares = infer_grid(cropped_warped)
    t_digits = get_digits(cropped_warped, digit_squares, 28)
    return t_digits, cropped_warped, digit_squares


def modify_input_for_model(images):
    """
    Modifies the images to the input shaped used by model
    """
    images = np.asarray(images, dtype=np.float32)
    images = images.reshape(images.shape[0], 28, 28, 1)
    images = images.astype('float32')
    images /= 255
    return images


def find_digits_from_images(images):
    """
    Uses the CNN model to predict images
    """
    t_digits = predict_with_model(images)
    t_digits = correct_for_empty(images, t_digits)
    return t_digits


def predict_with_model(images):
    modified_images = modify_input_for_model(images)
    model = keras.models.load_model("models/mnist_model")
    preds = model.predict_classes(modified_images)
    # Its bad with 1s and 7s, use OCR instead
    custom_config = r'--oem 1 --psm 10 outputbase digits'
    for i in range(len(preds)):
        if preds[i] == 7:
            try:
                d = pytesseract.image_to_string(
                    images[i], config=custom_config)
                if d.isdigit():
                    preds[i] = d

            except Exception:
                print("OCR fail")
    return preds


def correct_for_empty(images, t_digits):
    """
    If a slot contains white pixles above a certain threshold, mark it as 0 (no number contained)
    """
    for i in range(len(images)):
        num_white_px = np.sum(images[i] == 255)
        if num_white_px / (28*28) < 0.1:
            t_digits[i] = 0
    return t_digits


def write_solution(t_image, t_grid, t_solved_grid, boxes):
    print(solved_grid)
    print(boxes[0])
    boxes = np.reshape(boxes, (9, 9, 4), "F")
    print(boxes[0][0][0])
    for i in range(9):
        for j in range(9):
            if not t_grid[i][j]:
                org = (int(boxes[i][j][0])+13, int(boxes[i][j][3])-13)
                cv2.putText(
                    t_image, str(t_solved_grid[i][j]), org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    show_image(t_image, "solve")


if __name__ == "__main__":
    IMG_PATH = "imgs/sudoku7.jpg"  # Image to use - .jpg or .png
    sudoku = cv2.imread(IMG_PATH, 1)
    digits_images, cropped_warped, squares = find_sudoku(sudoku)
    digits = find_digits_from_images(digits_images)
    grid = np.reshape(digits, (9, 9), "F")

    solved_grid, found = solve_sudoku.solve(grid.copy())
    if found:
        write_solution(cropped_warped, grid, solved_grid, squares)
    else:
        print("Failed to find solution")
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # Close all windows
