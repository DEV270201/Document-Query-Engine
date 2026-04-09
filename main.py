from langchain_community.document_loaders import TextLoader, PyPDFLoader  # type: ignore
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from pathlib import Path as path
from langchain_chroma import Chroma # type: ignore
from datetime import datetime
from sentence_transformers import CrossEncoder # type: ignore
from langchain_ollama import ChatOllama # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore

cwd = path.cwd()
DATA_DIR = rf"{cwd}\data"
data_dir = path(DATA_DIR)
batch_size = 5

class VectorStorageManager:
    def __init__(self, db_path=rf"{cwd}\db\chromadb", embedding_model=None):
        self.db_path = db_path
        self.embedding_model = embedding_model

    def get_collection(self, collection_name=None):
        if collection_name is None:
            print("No collections specified...")
            return None
        
        if self.embedding_model is None:
            print("No embedding model specified...")
            return None
        
        return Chroma(
             collection_name=collection_name,
             embedding_function=self.embedding_model,
             persist_directory=self.db_path
        )


class EmbeddingsGenerator:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", model_args={'device': 'cpu'}, encode_args={}):
        self.model_name = model_name
        self.model_args = model_args
        self.encode_args = encode_args
        self.generator = None
        self._load_model()

    
    def _load_model(self):
        self.generator = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_args,
            encode_kwargs=self.encode_args
        )

class Retriever:
    def __init__(self, llm_model=None, db_manager:VectorStorageManager=None, reranker_model_name="BAAI/bge-reranker-base"):
        self.llm_model=llm_model
        self.db_manager=db_manager
        self.reranker_model_name=reranker_model_name
        self.reranker=self._load_reranker_model()

    
    def _load_reranker_model(self):
        print(f"Loading reranker model: {self.reranker_model_name}")
        return CrossEncoder(self.reranker_model_name, device='cpu')

    

    def get_top_results(self, k:int=2, collection_name:str="aws_db", user_query:str=None):
        query = user_query.strip()
        if not query:
            print(f"Failure | User query not specified ....")
            return []
        
        vector_db = self.db_manager.get_collection(collection_name=collection_name)

        if not vector_db:
            print("Failure | No such collection exists ....")
            return []
        
        # stage 1: initial chunk retrieval 
        initial_recall = k*2
        print(f"Fetching top {initial_recall} candidate chunks....")
        top_chunk_results = vector_db.similarity_search_with_score(query, k) # returns (chunk [page_content, metadata], similarity_score)

        for chunk,score in top_chunk_results:
            chunk.metadata['similarity_score'] = score

        print("Reranking Stage: Scoring candidates...")
        
        # Prepare pairs for the Cross-Encoder: [(Query, Chunk1), (Query, Chunk2), ...]
        pairs = [[query, chunk.page_content] for chunk,_ in top_chunk_results]
        
        # Get scores (higher is better)
        scores = self.reranker.predict(pairs)

        # Attach scores to the chunks and sort them
        for i, (chunk,_) in enumerate(top_chunk_results):
            chunk.metadata["rerank_score"] = float(scores[i])

        reranked_results = sorted(top_chunk_results, key=lambda x: x[0].metadata["rerank_score"], reverse=True)

        # Return only the top 'k' requested
        final_results = [result[0] for result in reranked_results]

        for i, chunk in enumerate(final_results):
            print(f"--- Reranked Chunk {i+1} ---")
            print(f"CHUNK: {chunk.page_content}") # Print snippet for brevity
            print(f"Metadata: {chunk.metadata}")
            print("\n")

        return final_results
    

class ResponseGenerator:
    def __init__(self, model_name="phi3:latest", temperature=0):
        self.model_name=model_name
        self.temperature=temperature
        self.llm=None
        self._load_model()

    def _load_model(self):
        print(f"Loading LLM {self.model_name} ....")
        self.llm=ChatOllama(model=self.model_name, temperature=self.temperature)
        print(f"Successfully connected to Ollama model: {self.model_name}")

    def generate_response(self, query, context):
        if not context:
            raise ValueError("Sorry, cannot process the request without context ....")
        
        if not self.llm:
            raise ValueError("LLM not connected ....")
        
        context_text = "\n\n".join([chunk.page_content for chunk in context])

        template = """
        You are a technical AWS assistant. Use the provided context to answer the question. 
        Do not add any fluff from your side. If the answer isn't in the context, honestly state that you don't know.
        If an user expresses their interests, take that into account and form your response effectively
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        try:
            # 3. Create a simple chain
            chain = prompt | self.llm | StrOutputParser()
            # 4. Generate the response
            for chunk in chain.stream({"context": context_text, "question": query}):
                print(chunk, end="", flush=True)
        except Exception as e:
            print("\n")
            print(f"Failure | {e}")
        return
    

class TextSpitter:
    def __init__(self):
        self.name="Text splitter class"

    def get_character_splitter(self, seperator_="\n\n", chunk_size_=1000, chunk_overlap_=200):
        return CharacterTextSplitter(
        separator=seperator_,
        chunk_size=chunk_size_,     # Maximum characters per chunk
        chunk_overlap=chunk_overlap_,   # Overlap between consecutive chunks
        length_function=len
    ) 

    def get_recursive_splitter(self, seperator_=["\n\n", "\n", " ", ""], chunk_size_=800, chunk_overlap_=300):
       return RecursiveCharacterTextSplitter(
        separators=seperator_, 
        chunk_size=chunk_size_,
        chunk_overlap=chunk_overlap_,
        length_function=len,
        is_separator_regex=False,
    ) 


def get_loader(file_path: path):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(file_path))
    elif ext == ".txt":
        return TextLoader(str(file_path), encoding="utf-8")
    else:
        return None
    


def generate_embeddings(db_manager=None):
    namespace = "aws_db3"
    vector_db = db_manager.get_collection(collection_name=namespace)

    if vector_db is None:
        print("No vector storage initialized .... quitting the app")
        return
    
    splitter = TextSpitter()
    text_splitter = splitter.get_character_splitter(seperator_="\n", chunk_overlap_=400)

    for file_path in data_dir.rglob("*"):

        if not file_path.is_file():
            continue

        file_name = file_path.name
        loader = get_loader(file_path)
        
        if not loader:
             print(f"Name: {file_name} | {file_path.suffix.lower()} extension not supported....")
             continue

        print(f"Name: {file_name} | Generating documents lazily ...")
        #1. parsing the document

        try:
            for doc_index, doc in enumerate(loader.lazy_load()):  
                metadata = {
                    "name": file_name,
                    "last_modified": datetime.now().isoformat(),
                    "doc_index": doc_index,
                    "extension": file_path.suffix.lower(),
                    "doc_type" : "Theory"
                }
                doc.metadata.update(metadata)
                current_batch = []

                #2. splitting up the documents
                print(f"Name: {file_name} | Splitting up the documents...")
                chunks = text_splitter.split_documents([doc])

                for index, chunk in enumerate(chunks):
                        #3. Embedding and storing the documents
                        current_batch.append(chunk)
                        if len(current_batch) >= batch_size:
                            print(f"Name: {file_name} | Pushing {len(current_batch)} chunks into {namespace} namespace...")
                            try:
                                vector_db.add_documents(current_batch)
                            except Exception as error:
                                print(f"Failure | Name: {file_name} | Embeddings failed to store | Doc index: {doc_index} | Start chunk index: {index - len(current_batch) + 1} | Batch size: {len(current_batch)}")
                            finally:
                                current_batch = []
                
                try:
                    if current_batch:
                        print(f"Name: {file_name} | Pushing {len(current_batch)} chunks into {namespace} namespace...")
                        vector_db.add_documents(current_batch)
                        current_batch = []
                except Exception as error:
                        print(f"Failure | Name: {file_name} | Embeddings failed to store | Doc index: {doc_index} | Start chunk index: {len(chunks) - len(current_batch)} | Batch size: {len(current_batch)}")
            
            print(f"Success | Name: {file_name} | Embeddings stored successfully!")
        
        except Exception as e:
            print(f"Failure | Name: {file_name} | Document generation failed ... | Reason: {e}")
        
    return


def retrieve_results(retrieve=None, user_query:str=None):
    if not user_query:
        print("Invalid request ....")
        raise ValueError("Error | User query empty ....")
    collection_name = input("Enter collection name: ")
    k = int(input("Enter K for top k results: "))
    
    results = retrieve.get_top_results(collection_name=collection_name, k=k, user_query=user_query)
    return results

def main():
    print("Loading up the embedding model.....")
    embeddings = EmbeddingsGenerator(encode_args={
        'normalize_embeddings': True
    })
    print("Embedding model initialized....")
    db_manager = VectorStorageManager(embedding_model=embeddings.generator)
    print("Loading retrieval class ....")
    retrieve = Retriever(db_manager=db_manager)
    print("Retriever class loaded successfully!")
    generator = ResponseGenerator(model_name="phi3:latest")

    while(True):
        print("Choose Options: ")
        print("1. Generate Embeddings")
        print("2. Query Database")
        print("3. Exit")

        user_input = int(input("Enter your options: "))

        match user_input:
            case 1:
                generate_embeddings(db_manager)
            case 2:
                try:
                    user_query = input("Enter your query: ")
                    user_query = user_query.strip()
                    # retrieval and reranking
                    context_docs = retrieve_results(retrieve, user_query)

                    #generate the llm response
                    print("Generating final response ....")
                    llm_response = generator.generate_response(context=context_docs,query=user_query)
                    # print("\n--- AI RESPONSE ---")
                    # print(llm_response)
                    # print("-------------------\n")
                except Exception as e:
                    print("Error: ", e)
                    print("\n")
            case _:        
                print("Thanks!")
                break

if __name__ == "__main__":
    main()
