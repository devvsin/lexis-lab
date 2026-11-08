from functools import lru_cache

import fitz
from paddleocr import PaddleOCR


@lru_cache(maxsize=1)
def get_ocr() -> PaddleOCR:
    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def ocr_page(page: fitz.Page) -> str:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    image = pixmap.tobytes("png")

    results = get_ocr().predict(image)

    lines: list[str] = []

    for result in results:
        result_data = result.json
        texts = result_data.get("rec_texts", [])

        lines.extend(
            text.strip()
            for text in texts
            if text.strip()
        )

    return "\n".join(lines)