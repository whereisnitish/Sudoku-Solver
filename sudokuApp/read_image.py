# import the necessary packages
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import imutils
import cv2
from keras.models import load_model
import os

from .solve_sudoku import predict_sudoku

model = load_model('digit_model.h5')

def find_puzzle(image, debug=False):

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (7, 7), 3)

	thresh = cv2.adaptiveThreshold(blurred, 255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	thresh = cv2.bitwise_not(thresh)

	if debug:
		cv2.imshow("Puzzle Thresh", thresh)
		cv2.waitKey(0)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	puzzleCnt = None
	
	for c in cnts:
		
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		if len(approx) == 4:
			puzzleCnt = approx
			break

	if puzzleCnt is None:
		raise Exception(("Could not find Sudoku puzzle outline. "
			"Try debugging your thresholding and contour steps."))

	if debug:

		output = image.copy()
		cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
		cv2.imshow("Puzzle Outline", output)
		cv2.waitKey(0)

	puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
	warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
	if debug:
	
		cv2.imshow("Puzzle Transform", puzzle)
		cv2.waitKey(0)

	return (puzzle, warped)

def extract_digit(cell, debug=False):

	thresh = cv2.threshold(cell, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	thresh = clear_border(thresh)

	if debug:
		cv2.imshow("Cell Thresh", thresh)
		cv2.waitKey(0)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	# if no contours were found than this is an empty cell
	if len(cnts) == 0:
		return None

	c = max(cnts, key=cv2.contourArea)
	mask = np.zeros(thresh.shape, dtype="uint8")
	cv2.drawContours(mask, [c], -1, 255, -1)

	(h, w) = thresh.shape
	percentFilled = cv2.countNonZero(mask) / float(w * h)

	if percentFilled < 0.03:
		return None

	digit = cv2.bitwise_and(thresh, thresh, mask=mask)
	# check to see if we should visualize the masking step
	if debug:
		cv2.imshow("Digit", digit)
		cv2.waitKey(0)
	# return the digit to the calling function
	return digit

def rewrite_img(cell,predicted_value,img_path):
    # Load the image
	image_size = (600, 600)
	file_name, file_extension = os.path.splitext(img_path)
	grid_size = (66, 66)

	# Create a blank white image
	puzzle_image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255

	# Define the font and other text parameters
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 0.9
	font_thickness = 2
	font_color = (0, 0, 0)# Yellow color in BGR

	# Define grid color
	grid_color = (0, 0, 0)  # Black color in BGR

	# Define the cell locations and predicted numbers
	cell_locations = cell
	predicted_solution = predicted_value

	# Write the numbers on the image and draw grids
	for i in range(len(cell_locations)):
		for j in range(len(cell_locations[i])):
			x1, y1, x2, y2 = cell_locations[i][j]
			number = predicted_solution[i][j]

			# Calculate text position for centering
			text_x = int((x1 + x2) / 2)
			text_y = int((y1 + y2) / 2)

			# Write number on the image
			cv2.putText(puzzle_image, str(number), (text_x, text_y), font, font_scale, font_color, font_thickness)

			# Draw grid around the number
			cv2.rectangle(puzzle_image, (x1, y1), (x2, y2), grid_color, 2)

	# Save the image with numbers locally
	result_path = os.path.join(f"{file_name}_solved{file_extension}")
	print(result_path)
	cv2.imwrite(result_path, puzzle_image)
	return f'{file_name}_solved'

def main(img_path):

	image = cv2.imread(img_path)
	image = imutils.resize(image, width=600)
	puzzleImage, warped = find_puzzle(image)


	board = np.zeros((9, 9), dtype="int")

	stepX = warped.shape[1] // 9
	stepY = warped.shape[0] // 9

	numbers = ''
	cellLocs = []
	# loop over the grid locations
	for y in range(0, 9):

		row = []
		for x in range(0, 9):
			startX = x * stepX
			startY = y * stepY
			endX = (x + 1) * stepX
			endY = (y + 1) * stepY
			row.append((startX, startY, endX, endY))

			cell = warped[startY:endY, startX:endX]
			digit = extract_digit(cell)
			if digit is not None:
				roi = cv2.resize(digit, (28, 28))
				roi = roi.astype("float") / 255.0
				roi = img_to_array(roi)
				roi = np.expand_dims(roi, axis=0)
				pred = model.predict(roi).argmax(axis=1)[0]
				numbers += str(pred)
				board[y, x] = pred
			else:
				numbers += '0'
		cellLocs.append(row)
	predicted_solution = predict_sudoku(numbers)
	result_path = rewrite_img(cellLocs,predicted_solution,img_path)
	return result_path



