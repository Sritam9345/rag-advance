from paddleocr import PaddleOCR
import cv2
import numpy as np
from collections import defaultdict
import numpy as np

print("creating")

ocr = PaddleOCR(
    lang="en"
)

print("created")

def ocr_logic(image_bytes):
    
    img = cv2.imdecode(
    np.frombuffer(image_bytes, np.uint8),
    cv2.IMREAD_COLOR
)
    
    result = ocr.predict(img)
    
    page_result = result[0]  

    texts = page_result["rec_texts"]
    boxes = page_result["rec_boxes"]   
    scores = page_result["rec_scores"]


    plain_text = "\n".join(texts)
    
    table_like_text = reconstruct_rows(
    page_result["rec_texts"],
    page_result["rec_boxes"]
)
    
    # print(page_result["rec_texts"])
    
    final_text = clean_ocr_table(table_like_text)
    
    print(len(final_text))
    
    return final_text
    
    
    

def reconstruct_rows(rec_texts, rec_boxes, y_tol=15):

    items = []

    for text, box in zip(rec_texts, rec_boxes):

        xmin, ymin, xmax, ymax = box

        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        items.append(
            (
                center_y,
                center_x,
                text
            )
        )

    items.sort(key=lambda x: (x[0], x[1]))

    rows = []

    current_row = []
    current_y = None

    for y, x, text in items:

        if current_y is None:
            current_y = y

        if abs(y - current_y) <= y_tol:

            current_row.append((x, text))

        else:

            current_row.sort()

            rows.append(
                " | ".join(
                    text for _, text in current_row
                )
            )

            current_row = [(x, text)]
            current_y = y

    if current_row:

        current_row.sort()

        rows.append(
            " | ".join(
                text for _, text in current_row
            )
        )

    return "\n".join(rows)



def clean_ocr_table(text):
    
    lines = []

    for line in text.split("\n"):

        line = " ".join(line.split())

        if len(line.strip()) > 0:
            lines.append(line)

    return "\n".join(lines)


