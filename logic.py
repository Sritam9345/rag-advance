import pdfplumber 
import re
from table_join import table_format
import io
import aiohttp
from langchain_text_splitters import RecursiveCharacterTextSplitter


async def pdf_checker(pdf):

    results = []
    
    chunks = []

    with pdfplumber.open(pdf) as file:

        emebed_directly = 0
        ocr = 0
        vision = 0

        async with aiohttp.ClientSession() as session:

            for index, page in enumerate(file.pages):

                raw_text = page.extract_text() or ""

                tables = page.extract_tables()

                table_text = table_format(tables)

                if len(raw_text) == 0:
                    text_coverage = 1
                else:
                    text_coverage = len(table_text) / len(raw_text)

                page_nums = re.findall(
                    r"\d+(?:\.\d+)?",
                    raw_text
                )

                table_nums = re.findall(
                    r"\d+(?:\.\d+)?",
                    table_text
                )

                if len(page_nums) == 0:
                    numeric_coverage = 1
                else:
                    numeric_coverage = len(table_nums) / len(page_nums)

                row_header_coverage = 0

                for table in tables:

                    if len(table) == 0:
                        continue

                    headers = len(table[0])
                    rows = len(table)

                    if rows > 0 and headers / rows > 1:
                        row_header_coverage += 1

                action = decide_action(
                    page=page,
                    tables=tables,
                    text_coverage=text_coverage,
                    numeric_coverage=numeric_coverage,
                    row_header_coverage=row_header_coverage
                )

                if action == "EMBED_DIRECTLY":
                    emebed_directly += 1
                    continue

                elif action == "OCR":
                    ocr += 1

                elif action == "VISION":
                    vision += 1

                if action == "OCR" or action == "VISION":
                    image_bytes = page_to_image(page)

                    data = aiohttp.FormData()

                    data.add_field(
                        "file",
                        image_bytes,
                        filename=f"page_{index}.png",
                        content_type="image/png"
                    )

                    async with session.get(
                        f"http://localhost:8000/{action}",
                        data=data
                    ) as response:

                        final_text = await response.json()
                else:
                    final_text = raw_text
                
                
                chunks.extend(buildChunking(final_text))
                
                results.append(
                    {
                        "page": index,
                        "action": action,
                        "result": final_text
                    }
                )

    print("EMBED_DIRECTLY", emebed_directly)
    print("OCR", ocr)
    print("VISION", vision)
    

    return chunks
            
            
        
def decide_action(page, tables, text_coverage, numeric_coverage, row_header_coverage):
    num_tables = len(tables)
    num_images = len(page.images)
    num_curves = len(page.curves)

    
    if num_images == 0 and num_tables == 0 and num_curves < 50:
        return "EMBED_DIRECTLY"

    
    if num_images > 0 or num_curves >= 50:
        return "VISION"

    
    if num_tables > 0:
       
        if numeric_coverage >= 0.8:
            return "EMBED_DIRECTLY"
       
        if (
            text_coverage < 0.60
            or numeric_coverage < 0.50
            or row_header_coverage < 0.50
        ):
            return "OCR"
        
        return "EMBED_DIRECTLY"

    
    return "EMBED_DIRECTLY"



def page_to_image(page) -> bytes:
    
    page_img = page.to_image(resolution=100)

    buffer = io.BytesIO()

    page_img.original.save(
        buffer,
        format="PNG"
    )

    img_bytes = buffer.getvalue()
    
    return img_bytes


def buildChunking(text: str)-> list[str]:
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
    )
    
    return splitter.split_text(text)
    
    

if __name__ == '__main__':
    pdf_checker()