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

def generate_legal_response(question, context_docs, contexto_practico=None):
    system_prompt = """
Eres un abogado especialista en derecho ecuatoriano. Tu tarea es responder EXCLUSIVAMENTE con base en los textos legales entregados a continuación. Está TERMINANTEMENTE PROHIBIDO utilizar conocimiento externo, suposiciones, interpretaciones o completar información más allá de lo provisto.

🧠 Objetivo general:
Redacta una respuesta útil, clara y jurídica que pueda ser comprendida tanto por ciudadanos sin formación legal como por abogados.

🫱 Empatía inicial:
Si la pregunta revela angustia, preocupación o un problema delicado (como cárcel, salud, familia, etc.), comienza con una frase empática y humana, como: “Entendemos lo difícil que puede ser esta situación…” o “Lamentamos lo ocurrido y con gusto le orientamos…”.

📘 Estructura obligatoria:
1. Da una respuesta clara y directa a la pregunta, explicando el contenido legal con palabras sencillas.
2. Cada afirmación debe mencionar de qué artículo y qué código o ley proviene, si aplica.
3. Incluye citas textuales relevantes del texto legal, incluso si están truncadas.
4. Finaliza siempre con la frase: “Me baso en [artículos citados]”.

⚠️ Reglas estrictas:
- NO cites artículos, códigos o leyes que no estén literalmente presentes en el contexto legal proporcionado.
- NO utilices jurisprudencia, doctrina, interpretación propia ni conocimiento externo.
- NO completes ideas que no estén expresamente contenidas en el texto legal.
- Si no hay normativa aplicable, responde exactamente: “No encontré normativa aplicable. No me baso en ningún artículo.”
"""

    context_text = "\nDOCUMENTOS LEGALES:\n" + "\n".join(
        f"{doc['codigo']} Art.{doc['articulo']}: {doc['texto'][:600]}"
        for doc in context_docs
    )

    if contexto_practico:
        context_text += f"\n\n🧾 Contexto práctico adicional: {contexto_practico}"

    response = openai_client.chat.completions.create(
        model=CONFIG["OPENAI_MODEL"],
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": f"{question}\n\n{context_text}"}
        ],
        temperature=CONFIG["TEMPERATURE"],
        max_tokens=CONFIG["MAX_TOKENS"]
    )

    respuesta = response.choices[0].message.content.strip()
    tokens_usados = response.usage.total_tokens if response.usage else 0
    return respuesta, tokens_usados



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
    prompt = (
    "Reformula esta respuesta práctica legal para que suene humana, empática, cercana y útil para alguien sin conocimientos jurídicos. Usa segunda persona.\n\n"
    "✅ Conserva obligatoriamente:\n"
    "- Enlaces web útiles como http://consultas.funcionjudicial.gob.ec\n"
    "- Instrucciones o pasos que sirvan a cualquier persona\n"
    "- Nombres de instituciones públicas\n"
    "- Referencias legales si las hay\n\n"
    "❌ Elimina o generaliza:\n"
    "- Datos personales (nombres, apellidos, cédulas)\n"
    "- Información individualizada como montos de pensión, sueldos, edades, fechas específicas\n\n"
    f"Texto original:\n{texto_practico}"
)

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
@app.route("/query", methods=["GET", "POST"])
def handle_query():
    try:
        question = request.args.get("question", "").strip() if request.method == "GET" else request.get_json().get("question", "").strip()
        if not question:
            return jsonify({"error": "Se requiere 'question'"}), 400

        # ========== CONTEXTO LEGAL ==========
        query_engine = index.as_query_engine(similarity_top_k=TOP_K_RESULTS)
        pinecone_response = query_engine.query(question)

        context_docs = []
        biografia_juridica = {"alta": [], "media": [], "baja": []}

        total_docs = len(pinecone_response.source_nodes)
        alta_limite = int(total_docs * 0.3)
        media_limite = int(total_docs * 0.6)

        for i, node in enumerate(pinecone_response.source_nodes):
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

        # ========== RESPUESTAS ==========
        respuesta_legal, tokens_usados = generate_legal_response(question, context_docs)

        index_respuestas_abogados = pc.Index("indice-respuestas-abogados")
        embedding = embed_model._get_query_embedding(question)
        similares = index_respuestas_abogados.query(vector=embedding, top_k=1, include_metadata=True)

        respuesta_practica_reformulada = None
        urls_extraidas = []

        if similares.get("matches"):
            raw_text = similares["matches"][0]["metadata"].get("respuesta_abogado", "")
            urls_extraidas = re.findall(r"http[s]?://\S+", raw_text)
            respuesta_practica_reformulada = obtener_respuesta_practica(question)
            for url in urls_extraidas:
                if url not in respuesta_practica_reformulada:
                    respuesta_practica_reformulada += f"\n🔗 Más información: {url}"

        # ========== UNIFICAR RESPUESTA ==========
        bloques = []

        if respuesta_practica_reformulada:
            bloques.append("📌 Recomendación práctica:\n" + respuesta_practica_reformulada.strip())

        bloques.append("⚖️ Fundamento legal:\n" + respuesta_legal.strip())

        return jsonify({
            "respuesta": "\n\n---\n\n".join(bloques),
            "biografia_juridica": biografia_juridica,
            "tokens_usados": {"total_tokens": tokens_usados}
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ============= ENDPOINT PRINCIPAL =============
@app.route("/test-contexto-practico", methods=["POST"])
def test_contexto_practico():
    try:
        data = request.get_json()
        question = data.get("question", "").strip() if data else ""
        if not question:
            return jsonify({"error": "Se requiere 'question'"}), 400

        index_respuestas_abogados = pc.Index("indice-respuestas-abogados")
        embedding = embed_model._get_query_embedding(question)

        similares = index_respuestas_abogados.query(
            vector=embedding,
            top_k=1,
            include_metadata=True
        )

        if not similares.get("matches"):
            return jsonify({"respuesta": "❌ No se encontró coincidencia."})

        match = similares["matches"][0]
        score = match["score"]
        idea = match["metadata"].get("respuesta_abogado", "")
        descripcion = match["metadata"].get("descripcion", "")

        contexto_practico = f'En casos como "{descripcion.lower()}" se suele actuar de la siguiente forma: {idea[:300]}...'

        return jsonify({
            "score": score,
            "descripcion": descripcion,
            "respuesta_abogado": idea,
            "contexto_practico": contexto_practico
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
