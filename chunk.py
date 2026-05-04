def chunkPdf(text, chunk_size=1000, overlap=200):
    # This function is used to chunk the pdf file into smaller pieces for better processing.
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
        # This mean :
         when i = 0, it will take text[0:1000]
         when i = 800, it will take text[800:1800]  
         when i = 1600, it will take text[1600:2600] and so on.
    return chunks