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

    response = openai_client.chat.completions.create(
        model=CONFIG["OPENAI_MODEL"],
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": f"{question}\n\n{context_text}"}
        ],
        temperature=CONFIG["TEMPERATURE"],
        max_tokens=CONFIG["MAX_TOKENS"]
    )

    return response.choices[0].message.content.strip()


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
        if request.method == "GET":
            question = request.args.get("question", "").strip()
        else:
            data = request.get_json()
            question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Se requiere 'question'"}), 400

        if any(kw in question.lower() for kw in MATH_KEYWORDS):
            return jsonify({
                "respuesta": "Lamento no realizar cálculos. Contacte al administrador.",
                "biografia_juridica": [],
                "tokens_usados": {"total_tokens": 0}
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
            respuesta_practica = obtener_respuesta_practica(question)
            texto_final = "No encontré normativa aplicable. No me baso en ningún artículo."
            if respuesta_practica:
                texto_final += f"\n\n🧑‍⚖️ Consejo práctico:\n{respuesta_practica}"

            return jsonify({
                "respuesta": texto_final,
                "biografia_juridica": [],
                "tokens_usados": {"total_tokens": 0}
            })

        # Crear contexto legal
        context_text = "\n".join(
            f"{doc['codigo']} Art.{doc['articulo']}: {doc['texto'][:600]}"
            for doc in context_docs
        )

        # Llamada a OpenAI para generar respuesta legal
        chat_response = openai_client.chat.completions.create(
            model=CONFIG["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": """Eres un abogado especialista en derecho ecuatoriano. Tu trabajo es responder únicamente con base en los documentos legales proporcionados. No debes inventar información, ni usar conocimientos externos.

Responde de forma profesional y estructurada:

1. Explicación legal clara y directa (basada exclusivamente en los documentos).
2. Lista de artículos aplicables (número y código).
3. Citas textuales relevantes del texto legal.
4. Cierra con: "Me baso en [artículos citados]".

⚠️ Si no encuentras la respuesta en los documentos, responde: "No encontré normativa aplicable. No me baso en ningún artículo."
"""}, 
                {"role": "user", "content": f"{question}\n\nDOCUMENTOS LEGALES:\n{context_text}"}
            ],
            temperature=CONFIG["TEMPERATURE"],
            max_tokens=CONFIG["MAX_TOKENS"]
        )

        respuesta_legal = chat_response.choices[0].message.content.strip()
        tokens_usados = chat_response.usage.total_tokens

        respuesta_practica = obtener_respuesta_practica(question)

        respuesta_final = respuesta_legal
        if respuesta_practica:
            respuesta_final += f"\n\n🧑‍⚖️ Consejo práctico:\n{respuesta_practica}"

        biografia_juridica = [
            {
                "codigo": doc["codigo"],
                "articulo": doc["articulo"],
                "texto": doc["texto"]
            }
            for doc in context_docs[:MAX_ARTICULOS_CON_TEXTO]
        ]

        return jsonify({
            "respuesta": respuesta_final,
            "biografia_juridica": biografia_juridica,
            "tokens_usados": {"total_tokens": tokens_usados}
        })

    except Exception as e:
        return jsonify({
            "error": f"Error: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
