import os
import re
import unicodedata
import traceback
import uuid

from openai import OpenAI
from flask import Flask, request, jsonify, send_from_directory
from llama_index import VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore
from pinecone import Pinecone
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import ServiceContext
from llama_index.schema import Document  # ✅ Document (una sola import)
from docx import Document as DocxDocument
import tiktoken

import requests
from bs4 import BeautifulSoup

# ============= CONFIGURACIÓN =============
CONFIG = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    "INDEX_NAME": os.getenv("INDEX_NAME"),
    "PINECONE_ENV": os.getenv("PINECONE_ENV"),
    "OPENAI_MODEL": "gpt-5-mini",
    "TEMPERATURE": 0.3,
    "MAX_TOKENS": 2000,
    "GOOGLE_SEARCH_API_KEY": os.getenv("GOOGLE_SEARCH_API_KEY"),
    "GOOGLE_CX": os.getenv("GOOGLE_CX"),
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

# ✅ Conteo de tokens robusto (evita KeyError con modelos nuevos)
def contar_tokens(texto, model_name="gpt-5-mini"):
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        # Fallbacks seguros si el modelo no está mapeado
        try:
            enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(texto or ""))

# ✅ Helper para compatibilidad gpt-5* (usa max_completion_tokens)
def create_completion(client, model, messages, temperature, max_tokens):
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if str(model).startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    return client.chat.completions.create(**kwargs)

# ============= LLAMA INDEX / PINECONE =============
pc = Pinecone(api_key=CONFIG["PINECONE_API_KEY"], environment=CONFIG["PINECONE_ENV"])
pinecone_index = pc.Index(CONFIG["INDEX_NAME"])

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",
    api_key=CONFIG["OPENAI_API_KEY"]
)

# Índice de contratos
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
Si la pregunta revela angustia, preocupación o un problema delicado (como cárcel, salud, familia, etc.), comienza con una frase empática y humana.

📘 Estructura obligatoria:
1. Respuesta clara y directa.
2. Cada afirmación debe mencionar de qué artículo y qué código o ley proviene, si aplica.
3. Incluye citas textuales relevantes del texto legal, incluso si están truncadas.
4. Finaliza: “Me baso en [artículos citados]”.

⚠️ Reglas estrictas:
- NO cites artículos que no estén en el contexto.
- NO uses jurisprudencia/doctrina externa.
- Si no hay normativa aplicable, responde: “No encontré normativa aplicable. No me baso en ningún artículo.”
""".strip()

    context_text = "\nDOCUMENTOS LEGALES:\n" + "\n".join(
        f"{doc['codigo']} Art.{doc['articulo']}: {doc['texto'][:600]}"
        for doc in context_docs
    )

    if contexto_practico:
        context_text += f"\n\n🧾 Contexto práctico adicional: {contexto_practico}"

    response = create_completion(
        openai_client,
        CONFIG["OPENAI_MODEL"],
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{question}\n\n{context_text}"}
        ],
        CONFIG["TEMPERATURE"],
        CONFIG["MAX_TOKENS"]
    )

    respuesta = response.choices[0].message.content.strip()
    tokens_usados = response.usage.total_tokens if response.usage else 0
    return respuesta, tokens_usados

# ============= RESPUESTA PRÁCTICA =============
def obtener_respuesta_practica(question, score=None):
    practical_index_name = "indice-respuestas-abogados"
    practical_index = pc.Index(practical_index_name)

    practical_vector_store = PineconeVectorStore(pinecone_index=practical_index)

    practical_index_instance = VectorStoreIndex.from_vector_store(
        vector_store=practical_vector_store,
        service_context=service_context
    )

    engine = practical_index_instance.as_query_engine(similarity_top_k=1)
    resultado = engine.query(question)

    if not resultado.source_nodes:
        return None

    texto_practico = resultado.response.strip()

    if score is not None and score < 0.75:
        introduccion = (
            "❗ La respuesta no responde directamente a la pregunta del usuario.\n"
            "- Introduce con una frase empática y de cautela.\n"
            "- Reformula el contenido original sin afirmar nada que no esté en el texto.\n"
        )
    else:
        introduccion = (
            "✅ La respuesta es clara y útil para la pregunta del usuario:\n"
            "- Reformúlala en tono claro, amable y profesional.\n"
        )

    prompt = (
        "Reformula esta respuesta práctica legal para que suene humana, empática y útil. Usa segunda persona. "
        "Evalúa si responde directamente a la pregunta.\n\n"
        f"🧑‍⚖️ Pregunta del usuario: \"{question}\"\n\n"
        f"{introduccion}\n"
        "🔒 Reglas:\n"
        "- Conserva enlaces útiles si existen.\n"
        "- NO agregues enlaces si no están.\n"
        "- Elimina datos sensibles.\n\n"
        f"Texto original:\n{texto_practico}"
    )

    reformulado = create_completion(
        openai_client,
        CONFIG["OPENAI_MODEL"],
        [
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
        embedding = embed_model._get_query_embedding(question)  # mantiene tu versión
        similares = index_respuestas_abogados.query(vector=embedding, top_k=1, include_metadata=True)

        respuesta_practica_reformulada = None
        if similares.get("matches"):
            match = similares["matches"][0]
            score = match.get("score", 0)
            if score >= 0.6:
                respuesta_practica_reformulada = obtener_respuesta_practica(question, score=score)

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

# ======================= DESCARGA =======================
@app.route("/descargar/<filename>")
def descargar_archivo(filename):
    return send_from_directory("archivos_temp", filename, as_attachment=True)

def guardar_docx_y_retornar_link(texto, host_url):
    filename = f"{uuid.uuid4()}.docx"
    filepath = os.path.join("archivos_temp", filename)
    os.makedirs("archivos_temp", exist_ok=True)

    doc = DocxDocument()
    for linea in texto.split('\n'):
        doc.add_paragraph(linea)
    doc.save(filepath)

    return f"{host_url}descargar/{filename}"

# ======================= GENERAR CONTRATO COMPLETO =======================
@app.route("/generar-contrato-completo", methods=["POST"])
def generar_contrato_completo():
    try:
        data = request.get_json()
        pregunta = data.get("pregunta", "").strip()

        if not pregunta:
            return jsonify({"error": "La pregunta es obligatoria"}), 400

        contrato_query_engine = contratos_index.as_query_engine(similarity_top_k=1)
        resultado_contrato = contrato_query_engine.query(pregunta)

        contrato_base = ""
        if resultado_contrato.source_nodes:
            contrato_base = resultado_contrato.source_nodes[0].node.text.strip()

        if contrato_base:
            prompt = f"""
Eres un abogado ecuatoriano experto en redacción de documentos legales. Adapta el siguiente modelo jurídico a la solicitud del usuario. Mantén estructura y estilo, personaliza el contenido.

📄 Solicitud del usuario:
{pregunta}

📑 Modelo de referencia:
{contrato_base}

✍️ Instrucciones:
- Solo el documento (sin explicaciones).
- Lenguaje jurídico claro.
- Usa campos genéricos: [NOMBRE], [FECHA], etc.
""".strip()

            tokens_prompt = contar_tokens(prompt, CONFIG["OPENAI_MODEL"])

            if tokens_prompt <= 3000:
                modelo = "gpt-5-mini"
                max_tokens_salida = 1000
            else:
                modelo = "gpt-4o"
                max_tokens_salida = 8000

            response = create_completion(
                openai_client,
                modelo,
                [
                    {"role": "system", "content": "Eres un abogado ecuatoriano experto en redacción legal."},
                    {"role": "user", "content": prompt}
                ],
                CONFIG["TEMPERATURE"],
                max_tokens_salida
            )

            texto = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens if response.usage else 0

            if tokens > 8000:
                url = guardar_docx_y_retornar_link(texto, request.host_url)
                return jsonify({
                    "respuesta": f"✅ Documento generado. <a href='{url}' target='_blank'>Descargar Word (.docx)</a>.",
                    "tokens_usados": {"total_tokens": tokens},
                    "biografia_juridica": None
                })

            return jsonify({
                "respuesta": texto,
                "tokens_usados": {"total_tokens": tokens},
                "biografia_juridica": None
            })

        # === Si no hay modelo base, usa contexto legal del índice principal ===
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
Eres un abogado ecuatoriano experto en redacción de documentos legales. Redacta el documento solicitado.

✍️ Reglas:
- Entrega el documento directamente (sin explicaciones).
- Lenguaje legal claro y preciso.
- Si aplica, estructura con cláusulas/artículos.
- Usa campos genéricos: [NOMBRE], [FECHA], [CANTIDAD], [CIUDAD].
- Solo usa artículos del contexto.

📄 Solicitud:
{pregunta}

📚 Contexto legal (usa solo esto):
{chr(10).join(contexto_legal)}

🧾 Al final agrega “Fundamento legal” con los artículos usados.
""".strip()

        response = create_completion(
            openai_client,
            CONFIG["OPENAI_MODEL"],
            [
                {"role": "system", "content": "Eres un abogado ecuatoriano experto en redacción legal."},
                {"role": "user", "content": prompt}
            ],
            CONFIG["TEMPERATURE"],
            CONFIG["MAX_TOKENS"] - 500
        )

        texto = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else 0

        if tokens > 8000:
            url = guardar_docx_y_retornar_link(texto, request.host_url)
            return jsonify({
                "respuesta": f"✅ Documento generado. <a href='{url}' target='_blank'>Descargar Word (.docx)</a>.",
                "tokens_usados": {"total_tokens": tokens},
                "biografia_juridica": biografia_juridica
            })

        return jsonify({
            "respuesta": texto,
            "tokens_usados": {"total_tokens": tokens},
            "biografia_juridica": biografia_juridica
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# ============= WEBSEARCH PRINCIPAL =============
@app.route("/envtest", methods=["GET"])
def env_test():
    return jsonify({
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "SEARCH_ENGINE_ID": os.getenv("SEARCH_ENGINE_ID")
    })

GOOGLE_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("GOOGLE_CX")

SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY")  # Asegúrate de definirlo en tu entorno

@app.route("/websearch", methods=["POST"])
def websearch():
    try:
        pregunta = request.get_json().get("pregunta", "").strip()
        if not pregunta:
            return jsonify({"error": "Se requiere 'pregunta'"}), 400

        # ===== Buscar en índice de respuestas prácticas =====
        index_respuestas_abogados = pc.Index("indice-respuestas-abogados")
        embedding = embed_model._get_query_embedding(pregunta)
        similares = index_respuestas_abogados.query(
            vector=embedding, top_k=1, include_metadata=True
        )

        mostrar_respuesta_practica = False
        mostrar_google = True
        respuesta_practica_html = ""

        score = 0
        if similares.get("matches"):
            match = similares["matches"][0]
            score = match.get("score", 0)

            if score >= 0.70:
                mostrar_respuesta_practica = True
                mostrar_google = False
            elif 0.60 <= score < 0.70:
                mostrar_respuesta_practica = True
                mostrar_google = True
            else:
                mostrar_google = True

            if mostrar_respuesta_practica:
                respuesta_practica = obtener_respuesta_practica(pregunta, score=score)
                if respuesta_practica:
                    respuesta_practica_html = (
                        "<h3>📌 Recomendación práctica</h3>"
                        f"<div class='chat-respuesta'>{respuesta_practica}</div>"
                    )
                    if mostrar_google:
                        respuesta_practica_html += "<hr>"

        respuesta_google_html = ""
        tokens = 0

        if mostrar_google:
            # ===== Buscar en Google =====
            search_url = "https://www.googleapis.com/customsearch/v1"
            search_params = {
                "key": GOOGLE_API_KEY,
                "cx": SEARCH_ENGINE_ID,
                "q": pregunta,
                "siteSearch": "gob.ec",
                "siteSearchFilter": "i"
            }
            search_resp = requests.get(search_url, params=search_params)
            search_resp.raise_for_status()
            resultados = search_resp.json().get("items", [])[:5]

            if not resultados:
                respuesta_google_html = "<h3>🌐 Respuesta basada en Google</h3><div class='chat-respuesta'>❌ No se encontraron páginas relevantes.</div>"
            else:
                contexto = "\n".join(
                    [f"Título: {item['title']}\nLink: {item['link']}" for item in resultados]
                )

                prompt = f"""
Eres un abogado experto. Un usuario te hace la siguiente pregunta:

\"{pregunta}\"

Google te ha mostrado los siguientes resultados:

{contexto}

Tareas:
1) Responde claro y directo en un párrafo.
2) Indica que debe realizar el proceso en fuentes oficiales.
3) Recomienda enlaces más útiles (por títulos) en orden de relevancia.

No inventes datos que no estén respaldados por los títulos mostrados.
""".strip()

                response = create_completion(
                    openai_client,
                    CONFIG["OPENAI_MODEL"],
                    [
                        {"role": "system", "content": "Eres un abogado experto que ayuda con trámites legales en Ecuador."},
                        {"role": "user", "content": prompt}
                    ],
                    CONFIG["TEMPERATURE"],
                    800
                )

                respuesta_ia_cruda = response.choices[0].message.content.strip()
                tokens = response.usage.total_tokens if response.usage else 0

                enlaces_html = []
                for i, item in enumerate(resultados, 1):
                    titulo = item.get("title", f"Fuente {i}").strip()
                    url = item.get("link", "#")
                    enlaces_html.append(f"<li>🔗 <strong>{titulo}</strong><br><a href='{url}' target='_blank'>{url}</a></li>")

                if enlaces_html:
                    respuesta_ia_cruda += "<br><br><ul class='fuentes-web'>" + "\n".join(enlaces_html) + "</ul>"

                respuesta_google_html = f"<h3>🌐 Respuesta basada en Google</h3><div class='chat-respuesta'>{respuesta_ia_cruda}</div>"

        return jsonify({
            "respuesta": respuesta_practica_html + respuesta_google_html,
            "tokens_usados": {"total_tokens": tokens},
            "biografia_juridica": ""
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# ============= TEST CONTEXTO PRÁCTICO =============
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

# ============= RUN =============
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
