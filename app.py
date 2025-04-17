import os
import re
import traceback
import unicodedata
from flask import Flask, request, jsonify
from openai import OpenAI
from llama_index import VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import ServiceContext
from pinecone import Pinecone
from dotenv import load_dotenv

# ======================= CONFIGURACIÓN =======================
load_dotenv()

CONFIG = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    "INDEX_NAME": os.getenv("INDEX_NAME"),
    "PINECONE_ENV": os.getenv("PINECONE_ENV"),
    "OPENAI_MODEL": "gpt-3.5-turbo",
    "TEMPERATURE": 0.3,
    "MAX_TOKENS": 1500
}

TOP_K_RESULTS = 20
MAX_DOCS_TO_OPENAI = 6
MAX_TEXT_CHARS = 600  # <--- Aquí limitamos a OpenAI

MATH_KEYWORDS = ['calcular', 'cálculo', 'fórmula', 'sumar', 'restar', 'multiplicar', 'dividir']

# ======================= UTILS =======================
def normalizar(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    ).lower()

# ======================= APP =======================
app = Flask(__name__)

pc = Pinecone(api_key=CONFIG["PINECONE_API_KEY"], environment=CONFIG["PINECONE_ENV"])
pinecone_index = pc.Index(CONFIG["INDEX_NAME"])
openai_client = OpenAI(api_key=CONFIG["OPENAI_API_KEY"])

vector_store = PineconeVectorStore(pinecone_index=pinecone_index, text_key="text")
embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=CONFIG["OPENAI_API_KEY"])
service_context = ServiceContext.from_defaults(embed_model=embed_model)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

# ======================= OPENAI CALL =======================
def generate_legal_response(question, context_docs):
    system_prompt = """
    Eres un abogado especialista en derecho ecuatoriano. Tu tarea es responder EXCLUSIVAMENTE con base en los textos legales entregados a continuación. Está TERMINANTEMENTE PROHIBIDO utilizar conocimiento externo, suposiciones, interpretaciones o completar información más allá de lo provisto.

    ⚖️ Modo de redacción:
    - Redacta con tono empático, profesional y claro.
    - Comienza con una introducción cordial, como: “Con gusto le indico que…” o “Respecto a su consulta…”.
    - Sé directo y fundamentado, evitando frases como “como experto legal…”.

    📚 Estructura requerida:
    1. Redacta una respuesta explicativa clara, pero cada afirmación o dato debe mencionar expresamente de qué artículo y código o ley proviene.
    2. Incluye al final una lista de los artículos citados: número y nombre del código o ley.
    3. Incluye citas textuales legales, incluso si están truncadas por el contexto.
    4. Finaliza siempre con la frase: “Me baso en [artículos citados]”.

    ⚠️ Reglas estrictas:
    - NO cites artículos, códigos o leyes que no estén literal y explícitamente incluidos en el contexto proporcionado.
    - NO utilices jurisprudencia, doctrina, interpretación propia ni sentido común.
    - NO completes ideas ni “corrijas” el texto legal aunque te parezca incompleto.
    - Si no hay normativa aplicable: responde exactamente “No encontré normativa aplicable. No me baso en ningún artículo.”
    """



    context_text = "\nDOCUMENTOS LEGALES:\n" + "\n".join(
        f"{doc['codigo']} Art.{doc['articulo']}: {doc['texto'][:MAX_TEXT_CHARS]}"
        for doc in context_docs[:MAX_DOCS_TO_OPENAI]
    )

    response = openai_client.chat.completions.create(
        model=CONFIG["OPENAI_MODEL"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{question}\n\n{context_text}"}
        ],
        temperature=CONFIG["TEMPERATURE"],
        max_tokens=CONFIG["MAX_TOKENS"]
    )

    return response.choices[0].message.content, response.usage.total_tokens

# ======================= MAIN ENDPOINT =======================
@app.route("/query", methods=["POST"])
def handle_query():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Se requiere 'question'"}), 400

        if any(kw in question.lower() for kw in MATH_KEYWORDS):
            return jsonify({"respuesta": "Lamento no realizar cálculos. Contacte al administrador."})

        match = re.search(r"art[ií]culo\s+(\d+)\s+del\s+c[oó]digo\s+([\w\s]+)", question.lower())
        articulo_buscado = match.group(1) if match else None
        codigo_buscado = normalizar(match.group(2)) if match else None

        query_engine = index.as_query_engine(similarity_top_k=TOP_K_RESULTS)
        pinecone_response = query_engine.query(question)

        context_docs = []
        biografia_juridica = {"alta": [], "media": [], "baja": []}

        total_docs = len(pinecone_response.source_nodes)
        alta_limite = int(total_docs * 0.3)
        media_limite = int(total_docs * 0.6)

        for i, node in enumerate(pinecone_response.source_nodes):
            score = getattr(node, 'score', 0.0)
            metadata = getattr(node.node, 'metadata', {})
            codigo = metadata.get('code', '')
            articulo = metadata.get('article', '')
            texto = getattr(node.node, 'text', '') or metadata.get("text", '')
            texto = texto.strip()

            doc_data = {"codigo": codigo, "articulo": articulo, "texto": texto}
            context_docs.append(doc_data)

            if i < alta_limite:
                biografia_juridica["alta"].append(doc_data)
            elif i < media_limite:
                biografia_juridica["media"].append(doc_data)
            else:
                biografia_juridica["baja"].append(doc_data)

        if not context_docs:
            return jsonify({"respuesta": "No encontré normativa aplicable. No me baso en ningún artículo."})

        respuesta, tokens_usados = generate_legal_response(question, context_docs)

        return jsonify({
            "respuesta": respuesta,
            "biografia_juridica": biografia_juridica,
            "tokens_usados": {"total_tokens": tokens_usados}
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
