from langchain_community.document_loaders import TextLoader, PyPDFLoader, PyMuPDFLoader  # type: ignore
from langchain_text_splitters import TextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from pathlib import Path as path
from langchain_chroma import Chroma # type: ignore
from datetime import datetime
from sentence_transformers import CrossEncoder # type: ignore
from langchain_ollama import ChatOllama # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from langchain_core.documents import Document # type: ignore
from transformers import AutoTokenizer # type: ignore
import json
import re
import uuid

cwd = path.cwd()
DATA_DIR = rf"{cwd}\resume"
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
    
class ParentChunkStorageRetriever:
    def __init__(self, storage_path=rf"{path.cwd()}\db\resume\parent_store.jsonl"):
        self.storage_path = storage_path
        self.data = self._load()
    
    def _load(self):
        data = {}
        p = path(self.storage_path)
        if p.exists():
            with open(p, 'r') as f:
                for line in f:
                    # {"id": "...", "text": "..."}
                    entry = json.loads(line)
                    data[entry['id']] = entry['text']
        return data
    
    def save_parent(self, parent_id, text):
        if parent_id not in self.data:
            self.data[parent_id] = text
            with open(self.storage_path, 'a') as f:
                json.dump({"id": parent_id, "text": text}, f)
                f.write('\n')

    def get_parent(self, parent_id):
        return self.data.get(parent_id, "")


class EmbeddingsGenerator:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", model_args={'device': 'cpu'}, encode_args={}):
        self.model_name = model_name
        self.model_args = model_args
        self.encode_args = encode_args
        self.generator = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        self.generator = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_args,
            encode_kwargs=self.encode_args
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def count_tokens(self,text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def get_character_splitter(self, seperator_="\n\n", chunk_size_=1000, chunk_overlap_=200):
        return CharacterTextSplitter(
        separator=seperator_,
        chunk_size=chunk_size_,     # Maximum characters per chunk
        chunk_overlap=chunk_overlap_,   # Overlap between consecutive chunks
        length_function=self.count_tokens
    ) 

    def get_recursive_splitter(self, seperator_=["\n\n", "\n", " ", ""], chunk_size_=450, chunk_overlap_=200):
       return RecursiveCharacterTextSplitter(
        separators=seperator_, 
        chunk_size=chunk_size_,
        chunk_overlap=chunk_overlap_,
        length_function=self.count_tokens,
        is_separator_regex=False,
    ) 
    
class ResumeStructuralSplitter(TextSplitter):
    
    SECTION_HEADERS = re.compile(
        r'\n(Summary|Education|Professional Experience|Projects|'
        r'Leadership & Certifications|Leadership|Certifications|'
        r'Technical Skills|Skills)\n',
        re.IGNORECASE
    )
    
    # Matches position before "Job Title | Company, Location"
    SUB_HEADERS = re.compile(r'\n(?=.+?\|.+?(?:,|\n))')

    def split_text(self, text: str) -> list[str]:
        chunks = []
        
        sections = self.SECTION_HEADERS.split(text)
        it = iter(sections)
        
        preamble = next(it)
        print("\n")
        print("preamble:     ", preamble)
        print("\n")

        preamble = preamble.strip()

        # the first chunk of the document will always contain the name of the person in case of resume 
        candidate_name = preamble[:preamble.index('\n')]

        print("Candidate Name: ", candidate_name)

        # if preamble:
        #     chunks.append(preamble)
        
        for header, content in zip(it, it):
            section_text = f"{header}\n{content.strip()}"
            
            if re.search(r'experience|projects?', header, re.IGNORECASE):
                sub_chunks = self.SUB_HEADERS.split(section_text)

                for i, sc in enumerate(sub_chunks):
                    if i == 0: #this would be the header itself
                        continue

                    if sc.strip():
                        # carries parent section name into every sub-chunk
                        chunks.append(
                            f"Candidate: {candidate_name}\n"
                            f"This section talks about: {header.strip()}\n"
                            f"{sc.strip()}"
                        )
            else:
                chunks.append(
                    f"Candidate: {candidate_name}\n"
                    f"This section talks about: {section_text}"
                )
        
        return chunks

class Retriever:
    def __init__(self, parent_store_retriever:ParentChunkStorageRetriever, llm_model=None, db_manager:VectorStorageManager=None, reranker_model_name="BAAI/bge-reranker-base"):
        self.llm_model=llm_model
        self.db_manager=db_manager
        self.reranker_model_name=reranker_model_name
        self.parent_retriever=parent_store_retriever
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
        top_chunk_results = vector_db.similarity_search_with_score(query, initial_recall) # returns (chunk [page_content, metadata], similarity_score)

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
        final_parent_chunks = []
        seen_parent_ids = set()

        for index, result_chunk in enumerate(reranked_results):
            chunk = result_chunk[0]

            print(f"--- Reranked Chunk {index+1} ---")
            print(f"CHUNK: {chunk.page_content}") # Print snippet for brevity
            print(f"Metadata: {chunk.metadata}")
            print(f"rerank score: {chunk.metadata['rerank_score']}")
            print("\n")

            # filtering out chunks which are of lower relevance in order to reduce hallucinations and provide crisp responses 
            if chunk.metadata['rerank_score'] >= 0.2:
                p_id = chunk.metadata.get("parent_id")
                if p_id:
                    # If we haven't already added this parent's context
                    if p_id not in seen_parent_ids:
                        # swap the child text for the full Parent text
                        parent_text = self.parent_retriever.get_parent(parent_id=p_id)
                        if parent_text:
                            chunk.page_content = parent_text
                            final_parent_chunks.append(chunk)
                            seen_parent_ids.add(p_id)
                else:
                    # Fallback for chunks without parents
                    final_parent_chunks.append(chunk)

            # Stop once we have enough parent context for the LLM
            if len(final_parent_chunks) >= k:
                break

        if len(final_parent_chunks) == 0:
             print(f"DEBUG | All chunks were filtered out (Top score was below 0.2). Sending empty context to LLM ....")

        return final_parent_chunks
    

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
        # if not context:
        #     raise ValueError("Sorry, cannot process the request without context ....")
        
        if not self.llm:
            raise ValueError("LLM not connected ....")
        
        context_text = "\n\n".join([chunk.page_content for chunk in context])

        template = """
You are a restricted document-query bot. 
1. If the Context section below is EMPTY or does not contain the answer, you MUST say exactly: "Information not found in local docs."
2. Do NOT use your own knowledge. 
3. Do NOT provide external links which is not present in the context.
        
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
   

def get_loader(file_path: path):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return PyMuPDFLoader(str(file_path))
    elif ext == ".txt":
        return TextLoader(str(file_path), encoding="utf-8")
    else:
        return None

def load_full_document(loader, file_name, file_path):
    pages = loader.load()
    # merge all the pages ..... in case of resume, this wont hurt since the resume are generally smaller in size
    full_text = "\n".join([page.page_content for page in pages])
    merged_doc = Document(
        page_content=full_text,
        metadata = {
            "source": str(file_path),
            "name": file_name,
            "total_pages": len(pages),
            "extension": file_path.suffix.lower(),
            "doc_type" : "Resume"
        }
    )

    return merged_doc

 
def generate_embeddings(parent_store_retriever:ParentChunkStorageRetriever, embeddings: EmbeddingsGenerator, db_manager:VectorStorageManager=None):
    namespace = "new_resume"
    vector_db = db_manager.get_collection(collection_name=namespace)

    if vector_db is None:
        print("No vector storage initialized .... quitting the app")
        return
    
    # using parent-child retriever system for better performing RAG system
    # parent chunk spliiter
    parent_splitter = ResumeStructuralSplitter()
    # child chunk splitter
    child_splitter = embeddings.get_recursive_splitter(chunk_size_=256, chunk_overlap_=50)

    for file_path in data_dir.rglob("tp_resume.pdf"):

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
                doc = load_full_document(loader=loader, file_name=file_name, file_path=file_path)
                parents = parent_splitter.split_documents([doc])
                current_batch = []

                for p_idx, p_doc in enumerate(parents):
                        print("=========")
                        print("parent chunk: ", p_doc.page_content)
                        print("=========")
                        parent_id = f"{file_name}_{uuid.uuid1()}_{p_idx}"
                        parent_store_retriever.save_parent(parent_id=parent_id, text=p_doc.page_content) #storing the parent chunk in the json file .... used for providing context to the LLM

                        #every document will have candidate name in the first line and header in the second one (except the first parent doc because it would be just name and location etc)
                        candidate_name=""
                        header_info=""
                        subheader_info=""
                        
                        #because the first parent document will always contain the information of the candidate since we have designed the splitter that way
                        #so skipping for that .... for now we arent adding that chunk in the vector db but will see later if we need it
                        # if p_idx > 0:

                        first = p_doc.page_content.index("\n")
                        second = p_doc.page_content.index("\n", first+1)
                        third = p_doc.page_content.index("\n", second+1)
                        candidate_name = p_doc.page_content[:first]
                        header_info = p_doc.page_content[first+1:second]

                        if re.search(r'experience|projects?', header_info, re.IGNORECASE):
                            subheader_info = p_doc.page_content[second+1:third]

                        #2. splitting up the documents
                        print(f"Name: {file_name} | Splitting up the documents...")
                        children = child_splitter.split_text(p_doc.page_content)

                        full_context = f"{candidate_name}\n{header_info}\n{subheader_info}"

                        for c_idx, child_text in enumerate(children):
                            # chunk = comprehend_section(chunk, SECTIONS, current_section)
                            if not child_text.startswith("Candidate:"):
                               child_text_with_context = f"{full_context}\n{child_text}" if full_context.strip() else child_text
                            else:
                              child_text_with_context = child_text
                                
                            child_doc = Document(
                                page_content=child_text_with_context,
                                metadata={
                                    **p_doc.metadata,
                                    "parent_id": parent_id, 
                                    "child_index": c_idx,
                                    "parent_idx": p_idx
                                }
                            )
                            
                            # 3. Embedding and storing the documents
                            current_batch.append(child_doc)
                            if len(current_batch) >= batch_size:
                                print(f"Name: {file_name} | Pushing {len(current_batch)} chunks into {namespace} namespace...")
                                try:
                                    vector_db.add_documents(current_batch)
                                except Exception as error:
                                    print(f"Failure | Name: {file_name} | Embeddings failed to store | Parent chunk index: {p_idx} | Start index: {c_idx - len(current_batch)} | Batch size: {len(current_batch)}")
                                finally:
                                    current_batch = []
                        
                        try:
                            if current_batch:
                                print(f"Name: {file_name} | Pushing {len(current_batch)} chunks into {namespace} namespace...")
                                vector_db.add_documents(current_batch)
                                current_batch = []
                        except Exception as error:
                                print(f"Failure | Name: {file_name} | Embeddings failed to store | Parent chunk index: {p_idx} | Start index: {len(children) - len(current_batch)} | Batch size: {len(current_batch)}")
            
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
    print("Retriever class loaded successfully!")
    generator = ResponseGenerator(model_name="phi3:latest")
    parent_store_retriever = ParentChunkStorageRetriever()
    print("Loading retrieval class ....")
    retrieve = Retriever(parent_store_retriever=parent_store_retriever,db_manager=db_manager)

    while(True):
        print("Choose Options: ")
        print("1. Generate Embeddings")
        print("2. Query Database")
        print("3. Exit")

        user_input = int(input("Enter your options: "))

        match user_input:
            case 1:
                generate_embeddings(parent_store_retriever, embeddings, db_manager)
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



    # this is my previous approach where i used to load one page at a time in order to reduce memory consumption 
    # breaks when the resume is multi page due to lost context across headers
    # for doc_index, doc in enumerate(loader.lazy_load()):  
    #             print("doc_index: ", doc_index)
    #             print("page_content: ", repr(doc.page_content))
    #             doc_metadata = {
    #                 "extension": file_path.suffix.lower(),
    #                 "doc_type" : "Theory"
    #             }

    #             parents = parent_splitter.split_documents([doc])
    #             current_batch = []

    #             for p_idx, p_doc in enumerate(parents):
    #                     print("=========")
    #                     print("parent chunk: ", p_doc.page_content)
    #                     print("=========")
    #                     parent_id = f"{file_name}_{doc_index}_{p_idx}"
    #                     parent_store_retriever.save_parent(parent_id=parent_id, text=p_doc.page_content) #storing the parent chunk in the json file .... used for providing context to the LLM

    #                     #2. splitting up the documents
    #                     print(f"Name: {file_name} | Splitting up the documents...")
    #                     children = child_splitter.split_text(p_doc.page_content)

    #                     for c_idx, child_text in enumerate(children):
    #                         # chunk = comprehend_section(chunk, SECTIONS, current_section)
    #                         child_doc = Document(
    #                                 page_content=child_text,
    #                                 metadata={
    #                                     **doc_metadata,
    #                                     **p_doc.metadata,
    #                                     "parent_id": parent_id, 
    #                                     "child_index": c_idx,
    #                                     "parent_idx": p_idx
    #                                 }
    #                             )
                            
    #                         # 3. Embedding and storing the documents
    #                     #     current_batch.append(child_doc)
    #                     #     if len(current_batch) >= batch_size:
    #                     #         print(f"Name: {file_name} | Pushing {len(current_batch)} chunks into {namespace} namespace...")
    #                     #         try:
    #                     #             vector_db.add_documents(current_batch)
    #                     #         except Exception as error:
    #                     #             print(f"Failure | Name: {file_name} | Embeddings failed to store | Doc index: {doc_index} | Parent chunk index: {p_idx} | Start index: {c_idx - len(current_batch)} | Batch size: {len(current_batch)}")
    #                     #         finally:
    #                     #             current_batch = []
                        
    #                     # try:
    #                     #     if current_batch:
    #                     #         print(f"Name: {file_name} | Pushing {len(current_batch)} chunks into {namespace} namespace...")
    #                     #         vector_db.add_documents(current_batch)
    #                     #         current_batch = []
    #                     # except Exception as error:
    #                     #         print(f"Failure | Name: {file_name} | Embeddings failed to store | Doc index: {doc_index} | Parent chunk index: {p_idx} | Start index: {len(children) - len(current_batch)} | Batch size: {len(current_batch)}")
            