import os
from openai import OpenAI
from flask import Flask, request, jsonify
from llama_index import VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore
from pinecone import Pinecone
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import ServiceContext
import traceback
import re
import unicodedata

# ============= CONFIGURACIÓN =============
CONFIG = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    "INDEX_NAME": os.getenv("INDEX_NAME"),
    "PINECONE_ENV": os.getenv("PINECONE_ENV"),
    "OPENAI_MODEL": "gpt-3.5-turbo",
    "TEMPERATURE": 0.3,
    "MAX_TOKENS": 2000
}

MATH_KEYWORDS = ['calcular', 'cálculo', 'fórmula', 'sumar', 'restar', 'multiplicar', 'dividir']
TOP_K_RESULTS = 25
SIMILARITY_THRESHOLD = 0.5
MAX_ARTICULOS_CON_TEXTO = 5

app = Flask(__name__)

# ============= UTILIDADES =============

def normalizar(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    ).lower()

# ============= LLAMA INDEX =============

pc = Pinecone(api_key=CONFIG["PINECONE_API_KEY"], environment=CONFIG["PINECONE_ENV"])
pinecone_index = pc.Index(CONFIG["INDEX_NAME"])

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index
)

embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",
    api_key=CONFIG["OPENAI_API_KEY"]
)

service_context = ServiceContext.from_defaults(embed_model=embed_model)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    service_context=service_context
)

openai_client = OpenAI(api_key=CONFIG["OPENAI_API_KEY"])

# ============= RESPUESTA LEGAL =============

def generate_legal_response(question, context_docs):
    system_prompt = """Eres un abogado especialista en derecho ecuatoriano. Tu trabajo es responder únicamente con base en los documentos legales proporcionados. No debes inventar información, ni usar conocimientos externos.

Responde de forma profesional y estructurada:

1. Explicación legal clara y directa (basada exclusivamente en los documentos).
2. Lista de artículos aplicables (número y código).
3. Citas textuales relevantes del texto legal.
4. Cierra con: "Me baso en [artículos citados]".

⚠️ Si no encuentras la respuesta en los documentos, responde: "No encontré normativa aplicable. No me baso en ningún artículo."
"""

    context_text = "\nDOCUMENTOS LEGALES:\n" + "\n".join(
        f"{doc['codigo']} Art.{doc['articulo']}: {doc['texto'][:600]}"
        for doc in context_docs
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

    return response.choices[0].message.content

# ============= RESPUESTA PRÁCTICA =============

def obtener_respuesta_practica(question):
    practical_index_name = "indice-respuestas-abogados"
    practical_index = pc.Index(practical_index_name)

    practical_vector_store = PineconeVectorStore(
        pinecone_index=practical_index
    )

    practical_index_instance = VectorStoreIndex.from_vector_store(
        vector_store=practical_vector_store,
        service_context=service_context
    )

    engine = practical_index_instance.as_query_engine(similarity_top_k=1)
    resultado = engine.query(question)

    if not resultado.source_nodes:
        return None

    texto_practico = resultado.response.strip()

    # Reformular con tono humano
    prompt = f"""
Reformula esta respuesta práctica legal para que suene humana, empática, cercana y útil para alguien sin conocimientos jurídicos. Usa segunda persona. No repitas textos literales ni artículos.

Texto original:
{texto_practico}
"""
    reformulado = openai_client.chat.completions.create(
        model=CONFIG["OPENAI_MODEL"],
        messages=[
            {"role": "system", "content": "Eres un asistente legal empático que habla en tono claro y humano."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=800
    )

    return reformulado.choices[0].message.content.strip()

# ============= ENDPOINT PRINCIPAL =============

@app.route("/query", methods=["POST"])
def handle_query():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Se requiere 'question'"}), 400

        if any(kw in question.lower() for kw in MATH_KEYWORDS):
            return jsonify({
                "respuesta": "Lamento no realizar cálculos. Contacte al administrador."
            })

        match = re.search(r"art[ií]culo\s+(\d+)\s+del\s+c[oó]digo\s+([\w\s]+)", question.lower())
        articulo_buscado = match.group(1) if match else None
        codigo_buscado = normalizar(match.group(2)) if match else None

        query_engine = index.as_query_engine(similarity_top_k=TOP_K_RESULTS)
        pinecone_response = query_engine.query(question)

        context_docs = []
        for node in pinecone_response.source_nodes:
            if hasattr(node, "score") and node.score < SIMILARITY_THRESHOLD:
                continue

            metadata = getattr(node.node, 'metadata', {})
            codigo = metadata.get('code', '')
            articulo = metadata.get('article', '')
            texto = metadata.get("respuesta_abogado", "")

            if articulo_buscado:
                if articulo == articulo_buscado and normalizar(codigo) == codigo_buscado:
                    context_docs.append({
                        "codigo": codigo,
                        "articulo": articulo,
                        "texto": texto
                    })
            else:
                context_docs.append({
                    "codigo": codigo,
                    "articulo": articulo,
                    "texto": texto
                })

        if not context_docs:
            return jsonify({
                "respuesta": "No encontré normativa aplicable. No me baso en ningún artículo.",
                "articulos_referenciados": [],
                "respuesta_practica_reformulada": obtener_respuesta_practica(question)
            })

        respuesta = generate_legal_response(question, context_docs)

        articulos_texto = context_docs[:MAX_ARTICULOS_CON_TEXTO]
        textos_adicionales = "\n\n📚 Otros artículos relacionados:\n"
        for doc in articulos_texto:
            textos_adicionales += f"{doc['codigo']} Art.{doc['articulo']}:\n{doc['texto']}\n\n"

        respuesta += textos_adicionales

        todos_articulos = list({
            f"{doc['codigo']} Art.{doc['articulo']}" for doc in context_docs
        })

        return jsonify({
            "respuesta": respuesta,
            "articulos_referenciados": todos_articulos,
            "respuesta_practica_reformulada": obtener_respuesta_practica(question)
        })

    except Exception as e:
        return jsonify({
            "error": f"Error: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
