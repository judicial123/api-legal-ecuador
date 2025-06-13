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

# Cargar el índice de Pinecone donde están los contratos
contratos_index = VectorStoreIndex.from_vector_store(
    PineconeVectorStore(
        pinecone_index=pc.Index("indice-contratos-legales"),
        text_key="text"
    ),
    service_context=ServiceContext.from_defaults(embed_model=embed_model)
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

def obtener_respuesta_practica(question, score=None):
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

    # Elegir introducción según el score
    if score is not None and score < 0.75:
        introduccion = (
            "❗ La respuesta no responde directamente a la pregunta del usuario.\n"
            "- Introduce con una frase como:\n"
            "  \"Es difícil indicarte si [reformula aquí la intención del usuario], en este caso es importante que un abogado experto te asesore. Sin embargo, puedo decirte que...\"\n"
            "- Reformula después el contenido original como referencia general.\n"
            "- No afirmes nada que no esté expresamente en el texto original.\n"
        )
    else:
        introduccion = (
            "✅ La respuesta es clara y útil para la pregunta del usuario:\n"
            "- Reformúlala sin alterar el mensaje, con un tono claro, amable y profesional.\n"
        )

    # Prompt final
    prompt = (
        "Reformula esta respuesta práctica legal para que suene humana, empática, cercana y útil para alguien sin conocimientos jurídicos. Usa segunda persona. Evalúa si responde o no directamente a la siguiente pregunta:\n\n"
        f"🧑‍⚖️ Pregunta del usuario: \"{question}\"\n\n"
        f"{introduccion}\n"
        "🔒 Reglas adicionales:\n"
        "- Conserva enlaces web útiles como http://consultas.funcionjudicial.gob.ec si están presentes en el texto original.\n"
        "- NO agregues enlaces si no están.\n"
        "- Elimina nombres propios, montos específicos, fechas y datos sensibles.\n\n"
        f"Texto original:\n{texto_practico}"
    )

    # Llamada a OpenAI
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

        if similares.get("matches"):
            match = similares["matches"][0]
            score = match.get("score", 0)
            if score >= 0.6:
                respuesta_practica_reformulada = obtener_respuesta_practica(question, score=score)
            else:
                respuesta_practica_reformulada = None

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


# ======================= GENERAR CONTRATO COMPLETO =======================
@app.route("/generar-contrato-completo", methods=["POST"])
def generar_contrato_completo():
    try:
        data = request.get_json()
        pregunta = data.get("pregunta", "").strip()

        if not pregunta:
            return jsonify({"error": "La pregunta es obligatoria"}), 400

        # Paso 1: Buscar contrato modelo en índice de contratos
        contrato_query_engine = contratos_index.as_query_engine(similarity_top_k=1)
        resultado_contrato = contrato_query_engine.query(pregunta)

        contrato_base = ""
        if resultado_contrato.source_nodes:
            contrato_base = resultado_contrato.source_nodes[0].node.text.strip()

        if contrato_base:
            # Si se encontró un contrato modelo
            prompt = f"""
Eres un abogado ecuatoriano experto en redacción de documentos legales. A continuación tienes un modelo jurídico que debes adaptar para responder a la solicitud del usuario. Mantén su estructura y estilo, pero personaliza el contenido según la petición.

📄 Solicitud del usuario:
{pregunta}

📑 Modelo de referencia:
{contrato_base}

✍️ Instrucciones:
- No incluyas explicaciones, solo el documento.
- Usa lenguaje jurídico claro.
- Usa campos genéricos como [NOMBRE], [FECHA], etc.
""".strip()
            response = openai_client.chat.completions.create(
                model=CONFIG["OPENAI_MODEL"],
                messages=[
                    {"role": "system", "content": "Eres un abogado ecuatoriano experto en redacción legal."},
                    {"role": "user", "content": prompt}
                ],
                temperature=CONFIG["TEMPERATURE"],
                max_tokens=CONFIG["MAX_TOKENS"] - 500
            )

            texto = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens

            return jsonify({
                "respuesta": texto,
                "tokens_usados": { "total_tokens": tokens },
                "biografia_juridica": None
            })

        # Paso 2: Si no hay contrato modelo, usar contexto legal tradicional
        query_engine = index.as_query_engine(similarity_top_k=10)
        resultado = query_engine.query(pregunta)

        contexto_legal = []
        biografia_juridica = {"alta": [], "media": [], "baja": []}

        total_docs = len(resultado.source_nodes)
        alta_limite = int(total_docs * 0.3)
        media_limite = int(total_docs * 0.6)

        for i, nodo in enumerate(resultado.source_nodes):
            meta = getattr(nodo.node, 'metadata', {})
            codigo = meta.get("code", "")
            articulo = meta.get("article", "")
            texto = getattr(nodo.node, 'text', '') or meta.get("text", '')
            texto = texto.strip()

            if codigo and articulo:
                contexto_legal.append(f"{codigo} Art. {articulo}: {texto[:500]}")
                doc = {"codigo": codigo, "articulo": articulo, "texto": texto}

                if i < alta_limite:
                    biografia_juridica["alta"].append(doc)
                elif i < media_limite:
                    biografia_juridica["media"].append(doc)
                else:
                    biografia_juridica["baja"].append(doc)

        prompt = f"""
Eres un abogado ecuatoriano experto en redacción de documentos legales. Vas a redactar un texto profesional, completo y jurídicamente válido, en respuesta a la solicitud del usuario.

✍️ Instrucciones estrictas:
- Redacta directamente el documento legal solicitado, sin explicaciones ni introducciones.
- Usa lenguaje legal claro y preciso, adecuado al sistema jurídico del Ecuador.
- Si el documento requiere estructura (contrato, demanda, reglamento, etc.), incluye numeración adecuada: cláusulas, artículos, incisos.
- Si el documento es breve (como una solicitud o escrito procesal), redacta en formato carta legal.
- Utiliza campos genéricos para datos personales: [NOMBRE], [FECHA], [CANTIDAD], [CIUDAD], etc.
- No agregues artículos legales inventados: usa solo los del contexto.

📄 Solicitud del usuario:
{pregunta}

📚 Contexto legal (solo puedes usar esto):
{chr(10).join(contexto_legal)}

🧾 Al final del documento, incluye una sección con el encabezado “Fundamento legal” y menciona los artículos usados.
""".strip()

        response = openai_client.chat.completions.create(
            model=CONFIG["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": "Eres un abogado ecuatoriano experto en redacción legal."},
                {"role": "user", "content": prompt}
            ],
            temperature=CONFIG["TEMPERATURE"],
            max_tokens=CONFIG["MAX_TOKENS"] - 500
        )

        texto = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens

        return jsonify({
            "respuesta": texto,
            "tokens_usados": { "total_tokens": tokens },
            "biografia_juridica": biografia_juridica
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
