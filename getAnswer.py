from vision import client
import aiohttp



async def getAnswer(userQuery):
    
    
    async with aiohttp.ClientSession() as session:
        
        async with session.post("http://localhost:8000/retrieve",params={"data": userQuery}) as response:
            
            data = await response.json()
            
    
    chunks = data['result'] 
        
    
    context = "\n\n".join(chunks) 
    
    print("got chunks")
    
    prompt = f"""
You are a document QA assistant.

Answer ONLY using the provided context.
If the answer is not present, say:
"I could not find that information."

Context:
{context}

Question:
{userQuery}
"""
    
    
    response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                prompt,
            ]
        )

    return response.text
    
    return "hi"