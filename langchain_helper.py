from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv


load_dotenv()
embeddings = OpenAIEmbeddings()

## Tools such as this 
def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    ## You can pull document loaders GCS file, S3 etc
    loader = YoutubeLoader.from_youtube_url(video_url)
    ## Saving into our transcript variable
    transcript = loader.load()
    ## You can split the massive transcript into smaller chunks
    ## overlap: First chunk will have 100 words that are in second chunk 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ## Splitting our youtube transcript document
    docs = text_splitter.split_documents(transcript)
    ## FIASS library is a library by facebook for our vector stores there are many of these
    ## We cannot send 1000 words to openai at a time as we will hit the limit
    ## We store these as vectors to get around this limit. 
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """
## Query will only pass the docs relevant to the query e.g new features in AI
    docs = db.similarity_search(query, k=k)
    ## We join the four docs 
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name="text-davinci-003")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )
    ## Where the magic happens everything is "chained" before we execute
    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs
