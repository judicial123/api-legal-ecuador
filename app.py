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

MATH_KEYWORDS = ['calcular', 'cálculo', 'fórmula', 'sumar', 'restar', 'multiplicar', 'dividir']
TOP_K_RESULTS = 15

# ======================= UTILIDADES =======================
def normalizar(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    ).lower()

# ======================= FLASK APP =======================
app = Flask(__name__)

# Inicializar Pinecone
pc = Pinecone(api_key=CONFIG["PINECONE_API_KEY"], environment=CONFIG["PINECONE_ENV"])
pinecone_index = pc.Index(CONFIG["INDEX_NAME"])
openai_client = OpenAI(api_key=CONFIG["OPENAI_API_KEY"])

# Inicializar LlamaIndex
vector_store = PineconeVectorStore(pinecone_index=pinecone_index, text_key="text")
embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=CONFIG["OPENAI_API_KEY"])
service_context = ServiceContext.from_defaults(embed_model=embed_model)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

# ======================= FUNCIÓN PRINCIPAL =======================
def generate_legal_response(question, context_docs):
    system_prompt = """Eres un abogado especialista en derecho ecuatoriano. Tu tarea es responder EXCLUSIVAMENTE con base en los documentos legales proporcionados a continuación. Está TERMINANTEMENTE PROHIBIDO utilizar conocimiento externo, incluso si es correcto o habitual.

⚠️ NO DEBES:
- Mencionar códigos, leyes, artículos o instituciones que no estén explícitamente incluidos en los documentos proporcionados.
- Hacer referencia indirecta a cuerpos legales ajenos (como “el Código de Procedimiento Civil”, “la ley”, “la jurisprudencia”, etc.) si no aparecen en el corpus recibido.
- Completar o ampliar la respuesta con información que no se encuentre literalmente en los textos legales entregados.

FORMATO:
1. Primer párrafo: Redacta como abogado con criterio profesional ecuatoriano. Sé claro, empático y fundamentado, pero basándote únicamente en los textos entregados. Inicia con un saludo cordial (por ejemplo: “Con gusto le indico que…” o “Respecto a su consulta…”), y responda directamente a la pregunta, sin frases como “como abogado especialista…”.
2. Parte legal:
- Lista de artículos citados (número y código)
- Citas textuales reales de los documentos
- Finaliza con: "Me baso en [artículos citados]"

⚠️ Si no encuentras normativa aplicable en los textos, responde exactamente:  
"No encontré normativa aplicable. No me baso en ningún artículo."
"""
    context_text = "\nDOCUMENTOS LEGALES:\n" + "\n".join(
        f"{doc['codigo']} Art.{doc['articulo']}: {doc['texto']}" for doc in context_docs
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

# ======================= FLUJO FLASK =======================
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

        # Ordenar resultados por score descendente
        ordenados = sorted(
            pinecone_response.source_nodes,
            key=lambda node: getattr(node, 'score', 0.0),
            reverse=True
        )

        total_docs = len(ordenados)
        alta_limite = int(total_docs * 0.3)
        media_limite = int(total_docs * 0.6)

        context_docs = []
        biografia_juridica = {"alta": [], "media": [], "baja": []}

        for i, node in enumerate(ordenados):
            score = getattr(node, 'score', 0.0)
            metadata = getattr(node.node, 'metadata', {})
            codigo = metadata.get('code', '')
            articulo = metadata.get('article', '')
            texto_completo = getattr(node.node, 'text', '') or metadata.get("text", '')
            texto = texto_completo.strip()

            print(f"🧠 Relevancia {score:.2f} → {codigo} Art.{articulo}")

            doc_data = {"codigo": codigo, "articulo": articulo, "texto": texto}

            if i < alta_limite:
                biografia_juridica["alta"].append(doc_data)
            elif i < media_limite:
                biografia_juridica["media"].append(doc_data)
            else:
                biografia_juridica["baja"].append(doc_data)

            context_docs.append(doc_data)

        if not context_docs:
            return jsonify({"respuesta": "No encontré normativa aplicable. No me baso en ningún artículo."})

        respuesta, tokens_usados = generate_legal_response(question, context_docs)

        return jsonify({
            "respuesta": respuesta,
            "biografia_juridica": biografia_juridica,
            "tokens_usados": { "total_tokens": tokens_usados }
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
