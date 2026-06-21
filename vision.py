from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


client = genai.Client(
    api_key=GEMINI_API_KEY
)

prompt = '''
You are processing a PDF page for a RAG system.

If the page contains:
- charts
- graphs
- diagrams
- infographics

Extract:

1. Chart title
2. Axis labels
3. Important trends
4. Numerical values visible
5. Key conclusions

Return plain text only.

Do not describe page layout.
Do not mention colors unless important.
'''



async def vision_response(image_bytes):
    
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png"
            )
        ]
    )

    return response.text