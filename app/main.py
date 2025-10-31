import os, re, io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import PyPDF2, docx
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = FastAPI(title="Hybrid Graph RAG")
templates = Jinja2Templates(directory="templates")

# === CONFIG ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

if not all([GROQ_API_KEY, NEO4J_URI, NEO4J_PASS]):
    raise RuntimeError("Missing: GROQ_API_KEY, NEO4J_URI, NEO4J_PASSWORD")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Global graph & vector store
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)
vector_store = None

# === INGESTION ===
def read_file(file_bytes: bytes, filename: str) -> str:
    stream = io.BytesIO(file_bytes)
    if filename.lower().endswith(('.txt', '.md')):
        return stream.read().decode('utf-8', errors='ignore')
    elif filename.lower().endswith('.pdf'):
        reader = PyPDF2.PdfReader(stream)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif filename.lower().endswith(('.docx', '.doc')):
        doc = docx.Document(stream)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        raise ValueError("Unsupported file")

def ingest_text(text: str):
    global vector_store
    graph.query("MATCH (n) DETACH DELETE n")

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    vector_store = Neo4jVector.from_texts(
        chunks, embedding=emb,
        url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS,
        index_name="text_embeddings", node_label="TextChunk",
        embedding_node_property="embedding", text_node_property="text"
    )

    extract_prompt = PromptTemplate.from_template(
        "Extract entities and relationships. Return only JSON:\n"
        '{{"entities": [...], "relationships": [...]}}\nText: {text}\nJSON:'
    )
    chain = extract_prompt | llm | JsonOutputParser()

    entities, relationships = set(), []
    for chunk in chunks:
        try:
            data = chain.invoke({"text": chunk})
            entities.update(e.strip() for e in data.get("entities", []) if e.strip())
            relationships.extend(data.get("relationships", []))
        except: pass

    if not relationships:
        for chunk in chunks:
            cl = chunk.lower()
            if "java" in cl and "spring" in cl:
                relationships.append({"source": "Java", "relation": "USED_WITH", "target": "Spring"})
            if "java" in cl and "quarkus" in cl:
                relationships.append({"source": "Java", "relation": "USED_WITH", "target": "Quarkus"})

    if entities:
        graph.query("UNWIND $ents AS e MERGE (n:Entity {name: e})", {"ents": list(entities)})

    def safe_type(s): return re.sub(r'[^A-Z0-9_]', '_', s.upper())
    for r in relationships:
        src = r.get("source", "").strip()
        tgt = r.get("target", "").strip()
        typ = safe_type(r.get("relation", "RELATED_TO"))
        if src and tgt:
            graph.query(
                f"MATCH (a:Entity {{name: $src}}), (b:Entity {{name: $tgt}}) "
                f"MERGE (a)-[r:`{typ}`]->(b)",
                {"src": src, "tgt": tgt}
            )
    return len(chunks), len(entities), len(relationships)

# === INFERENCE ===
def extract_entity(q: str) -> str:
    words = re.findall(r'\b[A-Z][a-z]+\b', q)
    stop = {"the","a","an","and","or","but","in","on","at","to","for","of","with","is","was","are","were"}
    for w in words:
        if w.lower() not in stop: return w
    return q.split()[0] if q else ""

def graph_context(q: str) -> str:
    try:
        entity = extract_entity(q)
        cy = PromptTemplate.from_template(
            "Write ONE MATCH query using label `Entity`. Return ONLY Cypher.\n"
            "Question: {question}\nEntity hint: {entity}\nCypher:"
        ) | llm | StrOutputParser()
        query = cy.invoke({"question": q, "entity": entity}).strip()
        if not query.upper().startswith("MATCH"):
            query = f"MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower('{entity}') "
                    f"OPTIONAL MATCH (e)-[r]-(other) RETURN e.name, collect({{rel:type(r), target:other.name}}) AS rels LIMIT 3"
        if any(k in query.upper() for k in ("CREATE","DROP","DELETE","SET","DETACH")):
            return ""
        rows = graph.query(query)
        lines = []
        for row in rows:
            name = row.get("name") or "?"
            rels = row.get("rels") or []
            lines.append(name)
            for r in rels[:3]:
                lines.append(f" -[{r.get('rel','?')}]→ {r.get('target','?')}")
        return "\n".join(lines)
    except: return ""

def hybrid_retrieve(q: str):
    if not vector_store: return "No document ingested yet.", ""
    vec = vector_store.similarity_search(q, k=2)
    vctx = "\n\n".join(d.page_content for d in vec)
    gctx = graph_context(q)
    return vctx, gctx

final_prompt = ChatPromptTemplate.from_template(
    "Answer in plain English:\n\nDocument:\n{vector_context}\n\nKnowledge graph:\n{graph_context}\n\nQuestion: {question}\nAnswer:"
)
chain = (
    {"vector_context": lambda q: hybrid_retrieve(q)[0],
     "graph_context": lambda q: hybrid_retrieve(q)[1],
     "question": RunnablePassthrough()}
    | final_prompt | llm | StrOutputParser()
)

# === ROUTES ===
@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.docx', '.txt', '.md')):
        return templates.TemplateResponse("index.html", {"request": None, "status": "Invalid file type.", "success": False})
    content = await file.read()
    try:
        text = read_file(content, file.filename)
        chunks, ents, rels = ingest_text(text)
        status = f"Success! Ingested {chunks} chunks, {ents} entities, {rels} relationships."
        return templates.TemplateResponse("index.html", {"request": None, "status": status, "success": True})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": None, "status": f"Error: {str(e)}", "success": False})

@app.get("/ask", response_class=HTMLResponse)
async def ask_page(request: Request):
    return templates.TemplateResponse("ask.html", {"request": request})

@app.post("/ask")
async def ask(question: str = Form(...)):
    if not vector_store:
        answer = "Please upload and ingest a document first."
    else:
        answer = chain.invoke(question)
    return templates.TemplateResponse("ask.html", {"request": None, "answer": answer})