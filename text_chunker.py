def chunkText(text, chunk_size=100, overlap=20):
    # This function is used to chunk the pdf file into smaller pieces for better processing.
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


